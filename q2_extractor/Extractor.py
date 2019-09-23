import zipfile
import yaml
import re
import io
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import h5py as h5
import tempfile
import copy
import arrow
import importlib
import inspect
import pkgutil

def scalar_constructor(loader, node):
    value = loader.construct_scalar(node)
    return value

yaml.add_constructor('!ref', scalar_constructor)
yaml.add_constructor('!no-provenance', scalar_constructor)
yaml.add_constructor('!color', scalar_constructor)
yaml.add_constructor('!cite', scalar_constructor)
yaml.add_constructor('!metadata', scalar_constructor)

def base_uuid(filename):
    regex = re.compile("[0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-" \
                       "[0-9a-fA-F]{4}\-[0-9a-fA-F]{12}")
    return regex.match(filename)[0]

def get_default_args(plugin_str, func_str):
    import qiime2
    pm = qiime2.sdk.PluginManager()
    if func_str in pm.plugins[plugin_str].methods:
        params = pm.plugins[plugin_str].methods[func_str].signature.parameters
        desc = pm.plugins[plugin_str].methods[func_str].description
    elif func_str in pm.plugins[plugin_str].visualizers:
        params = pm.plugins[plugin_str].visualizers[func_str].signature.parameters
        desc = pm.plugins[plugin_str].visualizers[func_str].description
    elif func_str in pm.plugins[plugin_str].pipelines:
        params = pm.plugins[plugin_str].pipelines[func_str].signature.parameters
        desc = pm.plugins[plugin_str].pipelines[func_str].description
    else:
        params = {}
        desc = "No description found"
    dat = {param: params[param].default for param in params}
    dat["step_description"] = desc
    return dat

#Deprecated, but I'm keeping this here just for fun
def manual_get_default_args(plugin_str, func_str):
    dat = None
    plugin_spec = importlib.util.find_spec(plugin_str)
    if plugin_spec:
        plugin = importlib.import_module(plugin_str)
        if "__path__" in vars(plugin):
            submods = [x[1] for x in pkgutil.iter_modules(plugin.__path__)]
        else:
            submods = []
    else:
        raise ValueError("Plugin %s not found" % (plugin_str,))
    if func_str not in vars(plugin):
        for submod in submods:
            mod_name = plugin_str + "." + submod
            mod_spec = importlib.util.find_spec(mod_name)
            if mod_spec:
                dat = get_default_args(mod_name, func_str)
                if dat is not None:
                    return dat
    else:
        try:
            signature = inspect.signature(getattr(plugin, func_str))
            dat = {
                k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            }
        except:
            # Try to get it from QIIME2 itself
            dat = get_default_args_auto(plugin_str, func_str)
    if dat is None:
        try:
            dat = get_default_args_auto(plugin_str, func_str)
        except:
            dat = None
    return dat

class Extractor(object):
    """This class attempts to extract the useful information
    from a QIIME2 artifact file.
    """
    def __init__(self, artifact_path):
        """
        """
        self.filename = artifact_path
        self.zfile = zipfile.ZipFile(artifact_path)
        self.infolist = self.zfile.infolist()
        self.base_uuid = base_uuid(self.infolist[0].filename)
        #First, hit up the lowest-level metadata.yaml
        xf = self.zfile.open(self.base_uuid + "/metadata.yaml")
        yf = yaml.load(xf, Loader=yaml.Loader)
        self.type = yf['type']
        self.format = yf['format']
        self.value_dict = {}
        #Next, hit up the action.yaml in the provenance folder
        #This is the provenance of THIS item
        xf = self.zfile.open(self.base_uuid + "/provenance/action/action.yaml")
        yf = yaml.load(xf, Loader=yaml.Loader)
        self.action_type = yf['action']['type']
        if self.action_type in ['method', 'pipeline', 'visualizer']:
            self.plugin = yf['action']['plugin'].split(":")[-1]
            self.action = yf['action']['action']
            self.parameters = yf['action']['parameters']
            self.inputs = yf['action']['inputs']
            self.plugin_versions = yf['environment']['plugins']
        elif self.action_type == 'import':
            self.format = yf['action']['format']
            self.action = 'import'
            self.plugin = 'qiime2'
            self.parameters = {}
            self.plugin_versions = yf['environment']['plugins']
        else:
            raise NotImplementedError("Action type '%s' not recognized." % (self.action_type,))
        #We can either get the input output info from transformers
        if 'transformers' in yf:
            if self.action_type in ['method', 'pipeline']:
                self.transforms = yf['transformers'] #autobots, roll out
                self.inputs = self.transforms['inputs']
                self.output = self.transforms['output']
            elif self.action_type == 'visualizer':
                self.transforms = yf['transformers']
                self.inputs = self.transforms['inputs']
                self.output = [{ 'to': yf['transformers']['inputs'][x][0]['to']} for x in yf['transformers']['inputs'] ]
            elif self.action_type == 'import':
                self.output = [{ 'to': yf['transformers']['output'][0]['to']}]
                self.inputs = {'import':[{'from':self.format}]}
        #Or if they aren't present (as occurs in pipelines), 
        #we have to go to metadata.yaml for outputs, and translate inputs
        else:
            self.inputs = {}
            for indict in yf['action']['inputs']:
                for item, uuid in indict.items():
                    if uuid is not None:
                        fmat = self.get_format_by_uuid(uuid)
                        self.inputs[item] = [{'from': fmat}]
                self.output = [{ 'to': yf['action']['output-name'] }]
        self.env = yf['environment']
        self.samples = None # Fetch this with get provenance if desired

    def get_provenance(self, current=True, upstream=True, include_input=True):
        actions = []
        latest_qiime_year = 2004
        latest_qiime_month = 1
        latest_qiime_minor = 0
        latest_rundate = arrow.Arrow(year=1979,month=1,day=1)
        if upstream:
            for fname in self.infolist:
                regex = re.compile("[0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-" \
                                   "[0-9a-fA-F]{4}\-[0-9a-fA-F]{12}/action/action.yaml")
                matches = regex.findall(fname.filename)
                if len(matches) >= 1:
                    actions.append("artifacts/" + matches[0])
        
        file_str = "sample_id\tvalue_type\tsequence_filename\tsequence_md5sum\n"
        if current:
            actions = actions + ["action/action.yaml"]
        samples = []
        result_stream = {}
        for actionyaml in actions:
            if actionyaml != "action/action.yaml":
                xf = self.zfile.open(self.base_uuid + "/provenance/artifacts/" + actionyaml.split("/")[1] + "/metadata.yaml")
                yf = yaml.load(xf, Loader=yaml.Loader)
                result_type = yf['type']
                result_format = yf['format']
            else:
                result_type = self.type
                result_format = self.format
            xf = self.zfile.open(self.base_uuid + "/provenance/" + actionyaml)
            yf = yaml.load(xf, Loader=yaml.Loader)
            rundate = yf['execution']['runtime']['start']
            rundate = arrow.get(rundate)
            if rundate > latest_rundate:
                latest_rundate = rundate
            if "version" in yf["environment"]["framework"]:
                qiime_version = yf['environment']['framework']['version']
                if latest_qiime_year < int(qiime_version.split(".")[0]):
                    latest_qiime_year = int(qiime_version.split(".")[0])
                if latest_qiime_month < int(qiime_version.split(".")[1]):
                    latest_qiime_month = int(qiime_version.split(".")[1])
                if latest_qiime_minor < int(qiime_version.split(".")[2]):
                    latest_qiime_minor = int(qiime_version.split(".")[2])
            res_uuid = actionyaml.split("/")[1]
            if res_uuid == "action.yaml":
                res_uuid = self.base_uuid
            if 'plugin' in yf['action']:
                parameters = [list(x.items())[0] for x in yf['action']['parameters']]
                inputs = [list(x.items())[0] for x in yf['action']['inputs']]
                inputs = [(x, y) for x,y in inputs if y is not None]
                plugin_name = yf['action']['plugin'].split(":")[-1]
                action = yf['action']['action']
                step = plugin_name + "__" + action 
                if res_uuid not in result_stream:
                    result_stream[res_uuid] = {"step_id": step}
                for key, value in parameters:
                    if key not in result_stream[res_uuid]:
                        result_stream[res_uuid][key] = [value]
                    else:
                        result_stream[res_uuid][key].append(value)
                for key, value in inputs:
                    if include_input:
                        if "upstream_result" not in result_stream[res_uuid]:
                            result_stream[res_uuid]["upstream_result"] = [value]
                        else:
                            result_stream[res_uuid]["upstream_result"].append(value)
                        if key not in result_stream[res_uuid]:
                            result_stream[res_uuid][key] = [value]
                        else:
                            result_stream[res_uuid][key].append(value)
                    else:
                        #Record the upstream step
                        upstream_uuid = value
                        if "upstream_step" not in result_stream[res_uuid]:
                            result_stream[res_uuid]["upstream_step"] = [upstream_uuid]
                        else:
                            result_stream[res_uuid]["upstream_step"].append(upstream_uuid)
            #It is import item, so we grab the manifest
            else:
                result_stream[res_uuid] = {"step_id": "qiime2_import"}
                fname_md5sums = yf['action']['manifest']
                filenames = [x['name'] for x in fname_md5sums]
                #If this is a MANIFESTed, import, we can get the sample_ids from it
                if include_input:
                    if ("MANIFEST" in filenames) and ("metadata.yml" in filenames):
                        samples = [x.split("_")[0] for x in filenames if x not in ["MANIFEST","metadata.yml"]]
                        self.samples = samples
                    for x in fname_md5sums:
                        if "input_filename" not in result_stream[res_uuid]:
                            result_stream[res_uuid]["input_filename"] = [x['name']]
                        else:
                            result_stream[res_uuid]["input_filename"].append(x['name'])
                        if "input_filename_md5sum" not in result_stream[res_uuid]:
                            result_stream[res_uuid]["input_filename_md5sum"] = [x['name']+": " + x['md5sum']]
                        else:
                            result_stream[res_uuid]["input_filename_md5sum"].append(x['name']+": "+x['md5sum'])
        
            result_stream[res_uuid]["result_type"] = [result_type]
        parameter_names = []
        for res_uuid in result_stream:
            if "upstream_step" in result_stream[res_uuid]:
                upstream_uuids = result_stream[res_uuid]["upstream_step"]
                upstream_steps = []
                for upstream_uuid in upstream_uuids:
                    if (upstream_uuid in result_stream) and \
                       ("step_id" in result_stream[upstream_uuid]) and \
                       (result_stream[upstream_uuid]["step_id"] not in upstream_steps):
                        upstream_steps.append(result_stream[upstream_uuid]["step_id"])
                result_stream[res_uuid]["upstream_step"] = upstream_steps

            initial_fields = list(result_stream[res_uuid].keys())
            for field in initial_fields:
                # If multiple values, duplicate columns
                if isinstance(result_stream[res_uuid][field], list):
                    orig_list = copy.deepcopy(result_stream[res_uuid][field])
                    for idx, val in enumerate(orig_list):
                        if idx == 0:
                            new_field = field
                        else:
                            new_field = field + ".%d" % (idx,)
                        parameter_names.append(new_field)
                        result_stream[res_uuid][new_field] = val
                else:
                    parameter_names.append(field)
        for res_uuid in result_stream:
            parameter_names.extend(list(result_stream[res_uuid].keys()))
        parameter_names = np.unique(parameter_names)
        plugin_columns = ["result_source","value_type", "value_target"] + parameter_names.tolist()
        plugin_table = pd.DataFrame.from_dict(result_stream, columns=plugin_columns, orient='index')
        plugin_table.index.name = "result_id"
        plugin_table.loc[:, "value_type"] = "parameter"
        if include_input:
            plugin_table.loc[:, "value_target"] = "result_id"
            plugin_table["value_target.1"] = "step_id"
            plugin_table.columns = [x if "value_target" not in x else "value_target" for x in plugin_table.columns ]
        else:
            plugin_table.loc[:, "value_target"] = "step_id"
        plugin_table.loc[:, "result_source"] = "qiime2"
#        plugin_table.loc[:, "analysis_id"] = "QIIME2 Run, " + latest_rundate.format("MMMM YYYY") + ", version %d.%d.%d" % (latest_qiime_year, latest_qiime_month, latest_qiime_minor)
#        plugin_table.loc[:, "analysis_date"] = latest_rundate.format("DD/MM/YYYY")
        for idx, sample in enumerate(samples):
            if idx == 0:
                new_field = "sample_id"
            else:
                new_field = "sample_id.%d" % (idx,)
            plugin_table.loc[:, new_field] = sample
        return plugin_table

    def get_result(self, *args, **kwargs):
        return self.get_provenance(*args, **kwargs)

    def get_samples(self):
        if self.samples is None:
            self.get_provenance()
        return self.samples

    def get_format_by_uuid(self, uuid):
        xf = self.zfile.open(self.base_uuid + "/provenance/artifacts/" + uuid + "/metadata.yaml")
        yf = yaml.load(xf, Loader=yaml.Loader)
        return yf['format']

    def extract_data(self):
        #TODO: Subclass this out into a TypeParser or something for better organization
        #Defines the functions for each QIIME artifact type, and outputs a Python object
        if self.type == 'SampleData[DADA2Stats]':
            #Output: pandas DataFrame
            data_file = self.base_uuid + "/data/stats.tsv"
            xf = self.zfile.open(data_file)
            tf = pd.read_csv(xf, sep="\t", comment = "#")
            return tf
        elif self.type == 'FeatureTable[Frequency]':
            #Output: pandas DataFrame
            data_file = self.base_uuid + "/data/feature-table.biom"
            with tempfile.NamedTemporaryFile() as temp_file:
                x=self.zfile.read(data_file)
                temp_file.write(x)
                tf = h5.File(temp_file.name)
                data = tf['observation/matrix/data'][:]
                indptr = tf['observation/matrix/indptr'][:]
                indices = tf['observation/matrix/indices'][:]
                sparse_mat = csr_matrix((data,indices,indptr))
                dense_mat = pd.DataFrame(sparse_mat.todense())
                dense_mat.rename(dict(zip(dense_mat.columns.values,tf['sample/ids'][:])), 
                                 inplace=True, axis='columns')
                dense_mat.rename(dict(zip(dense_mat.index.values,tf['observation/ids'][:])), 
                                 inplace=True, axis='index')
                return dense_mat
        elif self.type == 'FeatureData[Taxonomy]':
            data_file = self.base_uuid + "/data/taxonomy.tsv"
            xf = self.zfile.open(data_file)
            tf = pd.read_csv(xf, sep="\t")
            return tf
        elif self.type == 'PCoAResults':
            #This file is the dumbest to parse
            #It's a bunch of tables stacked on top of one another in ASCII
            data_file = self.base_uuid + "/data/ordination.txt"
            xf = self.zfile.open(data_file)
            nsamples = pd.read_csv(xf, sep="\t", skiprows=0, 
                                   nrows=1, header=None).loc[0][1]
            xf = self.zfile.open(data_file)
            eigvals = pd.read_csv(xf, sep="\t", skiprows=1, nrows=1, header=None)
            xf = self.zfile.open(data_file)
            assert pd.read_csv(xf, sep="\t", skiprows=2, 
                               nrows=1, header=None).loc[0][0] == 'Proportion explained'
            xf = self.zfile.open(data_file)
            prop_explained = pd.read_csv(xf, sep="\t", skiprows=4, nrows=1, header=None)
            xf = self.zfile.open(data_file)
            principal_coords = pd.read_csv(xf, sep="\t", skiprows=9, 
                                           nrows=nsamples, header=None, index_col=0)
            prop_explained.rename(index={0:"Proportion explained"}, inplace=True)
            eigvals.rename(index={0:"Eigenvalues"}, inplace=True)
            return {'eigenvalues': eigvals, 'proportion_explained': prop_explained, 
                    'coordinates': principal_coords}
        elif self.type == 'Phylogeny[Rooted]':
            import ete3
            data_file = self.base_uuid + "/data/tree.nwk"
            xf = self.zfile.open(data_file)
            tree = ete3.Tree(xf.read().decode(), format=1)
            return tree
        else:
            raise NotImplementedError("Type '%s' not yet implemented." % (self.type,))
   
    def _init_value_table(self):
        # Globally true settings for all artifacts
        step = self.plugin + "__" + self.action 
        global_settings = {"result_id": self.base_uuid,
                           "result_type": self.type,
                           "result_source": "qiime2",
                           "step_id": step,
                           "value_type": "metadata",
                           "value_target": "result_id"}
        self.value_dict = {0: global_settings}
        self.valtab_index = 1

    def _add_value(self, values, link_objs=[], value_type="measure"):
        #Duplicate the rows above
        self.value_dict[self.valtab_index] = copy.deepcopy(self.value_dict[0])
        for value_name, value in values:
            self.value_dict[self.valtab_index][value_name] = value
        for field_info in link_objs:
            field_name = field_info[0]
            original_field = field_name
            suffidx = 1
            while (field_name in self.value_dict[self.valtab_index]):
                field_name = field_info[0] + ".%d" % (suffidx,)
                suffidx += 1
            self.value_dict[self.valtab_index][field_name] = field_info[1]
            vfname = "value_target"
            idx = 1
            found=False
            while vfname in self.value_dict[self.valtab_index]:
                if self.value_dict[self.valtab_index][vfname] == original_field:
                    found=True
                    break
                else:
                    vfname = "value_target.%d" % (idx,)
                    idx += 1
            if not found:
                self.value_dict[self.valtab_index][vfname] = original_field
        self.value_dict[self.valtab_index]["value_type"] = value_type
        self.valtab_index += 1

    def get_values(self):
        self._init_value_table()
               # We need to add sample ID and 
        if self.type == 'SampleData[DADA2Stats]':
            data = self.extract_data()
            for row in data.index:
                sample = data.loc[row]['sample-id']
                index_names = ["input", "filtered","denoised","merged","non-chimeric"]
                value_names = ["input_sequence_count", "filtered_sequence_count", "denoised_sequence_count", "merged_sequence_count", "nonchimeric_sequence_count"]
                values = [data.loc[row][name] for name in index_names]
                self._add_value(zip(value_names, values),
                                [("sample_id", sample)])
        elif self.type == 'PCoAResults':
            data = self.extract_data()
            coords = data['coordinates']
            prop_exp = data['proportion_explained']
            # Set this to be variable?
            for x in coords.index:
                self._add_value([("pcoa_coord_%d" % (pc,), coords.loc[x][pc]) for pc in [1,2,3]],
                                [("sample_id", x)])
            self._add_value([("pcoa_proportion_explained_%d" % (pc,), prop_exp.loc["Proportion explained"][pc]) for pc in [1,2,3]])
        elif self.type == 'FeatureTable[Frequency]':
            table_data = self.extract_data()
            feature_names = table_data.index.tolist()
            sample_names = table_data.columns.tolist()
            sample_abundances = table_data.sum()
            feature_abundances = table_data.sum(axis=1)
            total_sequences = table_data.sum().sum()
            self._add_value([("sequence_count",
                            total_sequences)])
            for sample, abund in zip(sample_names, sample_abundances):
                self._add_value([("sequence_count",abund)],
                                [("sample_id", sample)])
            for feat, abund in zip(feature_names, feature_abundances):
                self._add_value([("sequence_count",
                                abund)],
                                [("feature_id", feat)])
            for sample in sample_names:
                for feat in feature_names:
                    abund = table_data.loc[feat][sample]
                    if (abund > 0):
                        self._add_value([("sequence_count",
                                    table_data.loc[feat][sample])],
                                    [("feature_id", feat),
                                     ("sample_id", sample)])
        elif self.type == 'FeatureData[Taxonomy]':
            data = []
            tax_data = self.extract_data()
            for index, row in tax_data.iterrows():
                self._add_value([("taxonomic_classification",
                                  row['Taxon']),
                                 ("taxonomic_confidence",
                                  row['Confidence']),
                                 ("feature_annotation",
                                  "taxonomic_classification")],
                                [("feature_id",
                                  row['Feature ID'])])
        elif self.type == 'Phylogeny[Rooted]':
            data = self.extract_data()
            self._add_value([("newick_string", data.write())],
                            [("sample_id", x) for x in self.get_samples()] + [("feature_id", x.name) for x in data.get_leaves()])
        return pd.DataFrame.from_dict(self.value_dict, orient='index')

    def __str__(self):
        o_str = "Artifact: %s\n" % (self.filename,)
        o_str += "Type: %s\n" % (self.type,)
        if self.action_type in ['method','pipeline']:
            o_str += "Result of action: %s\n" % (self.action,)
            param_str = ", ".join(["%s:%s" % (list(x.keys())[0], 
                                              list(x.values())[0]) for x in self.parameters])
            o_str += "Performed by plugin '%s' with parameters: %s \n" % (self.plugin, param_str)
            plugin_vers = zip(self.plugin_versions.keys(), 
                              [x['version'] for x in self.plugin_versions.values()])
            plugin_str = ", ".join(["%s (%s)" % (x[0],x[1]) for x in plugin_vers])
            o_str += "Loaded plugins and versions: %s" % (plugin_str,)
        return o_str
