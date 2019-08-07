import zipfile
import yaml
import re
import io
import pandas as pd
from scipy.sparse import csr_matrix
import h5py as h5
import tempfile
from sklearn.preprocessing import normalize

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

class q2Extractor(object):
    """This class attempts to extract the useful information
    from a QIIME2 artifact file.
    """
    # Types covered:
    #  SampleData[Dada2Stats]
    #  FeatureTable[Frequency]
    #  FeatureTable[Taxonomy]
    #  PCoAResults
    # Types needed:
    #  
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
        
        #Next, hit up the action.yaml in the provenance folder
        #This is the provenance of THIS item
        xf = self.zfile.open(self.base_uuid + "/provenance/action/action.yaml")
        yf = yaml.load(xf, Loader=yaml.Loader)
        self.action_type = yf['action']['type']
        if self.action_type in ['method', 'pipeline', 'visualizer']:
            self.plugin = yf['action']['plugin']
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

    def get_provenance(self):
        provenance_actions = []
        for fname in self.infolist:
            regex = re.compile("[0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-" \
                               "[0-9a-fA-F]{4}\-[0-9a-fA-F]{12}/action/action.yaml")
            matches = regex.findall(fname.filename)
            if len(matches) >= 1:
                provenance_actions.append(matches[0])
        
        plugin_str = "pipeline_step_id\tpipeline_step_action\tpipeline_step_parameter_id\tpipeline_step_parameter_value\n"
        file_str = "replicate_name\tfilename\tmd5sum\n"
        for actionyaml in provenance_actions:
            xf = self.zfile.open(self.base_uuid + "/provenance/artifacts/" + actionyaml)
            yf = yaml.load(xf, Loader=yaml.Loader)
            if 'plugin' in yf['action']:
                parameters = [list(x.items())[0] for x in yf['action']['parameters']]
                plugin_name = yf['action']['plugin'].split(":")[-1]
                action = yf['action']['action']
                for key, value in parameters:
                    plugin_str += "%s\t%s\t%s\t%s\n" % (plugin_name, action, key, value)
            #It is import item, so we grab the manifest
            else:
                fname_md5sums = yf['action']['manifest']
                for x in fname_md5sums:
                    file_str+= "%s\t%s\t%s\n" % (x['name'].split("_")[0] if ".fastq.gz" in x['name'] else "NA", 
                                                 x['name'], 
                                                 x['md5sum'])
        return plugin_str, file_str

    #Return the names of the replicates (i.e., fastq.gz filenames) involved in creating the artifact
    def get_replicates(self):
        file_prov = self.get_provenance()[1]
        return pd.unique(
                pd.read_csv(io.StringIO(self.get_provenance()[1]), 
                            sep="\t")["replicate_name"].dropna()).tolist()

    def get_format_by_uuid(self, uuid):
        xf = self.zfile.open(self.base_uuid + "/provenance/artifacts/" + uuid + "/metadata.yaml")
        yf = yaml.load(xf, Loader=yaml.Loader)
        return yf['format']

    def extract_data(self):
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
    
    def extract_measures(self):
        #Extracts measures in format (name, description, type, value, target, target_names)
        if self.type == 'SampleData[DADA2Stats]':
            pass
        elif self.type == 'PCoAResults':
            pass
        elif self.type == 'FeatureTable[Frequency]':
            data = []
            table_data = self.extract_data()
            feature_names = table_data.index.tolist()
            replicate_names = table_data.columns.tolist()
            rep_abundances = table_data.sum()
            feature_abundances = table_data.sum(axis=1)
            total_sequences = table_data.sum().sum()
            data = [('total_sequences', 
                      "Total sequences from a Feature Table", 
                      "Int", 
                      total_sequences, 
                      "BiologicalReplicate", 
                      self.get_replicates())]
            data.extend([('replicate_abundance',
                     "Total abundance for a replicate",
                     "Int",
                     abundance,
                     "BiologicalReplicate",
                     [rep_name]) for rep_name, abundance in rep_abundances.iteritems()])
            data.extend([('feature_abundance',
                    "Total abundance for a feature",
                    "Int",
                    abundance,
                    "Feature",
                    [feat_name]) for feat_name, abundance in feature_abundances.iteritems()])
            [data.extend([('feature_sample_abundance', 
                       "Abundance of a feature in a replicate",
                       "Int",
                       table_data.loc[feature, replicate],
                       "FeatureReplicate",
                       [(feature, replicate)]) for replicate in replicate_names if table_data.loc[feature, replicate] > 0.0]) for feature in feature_names]
        elif self.type == 'FeatureData[Taxonomy]':
            data = []
            tax_data = self.extract_data()
            return [("taxonomic_classification", 
                     "Taxonomic classification of a feature", 
                     "Str", 
                     row['Taxon'], 
                     "Feature",
                     [row['Feature ID']]) for index, row in tax_data.iterrows()]
        elif self.type == 'Phylogeny[Rooted]':
            data = [("newick_string", "Newick string representation of a phylogeny",
                     "Str", self.extract_data().write(), 
                     'BiologicalReplicate', self.get_replicates())]
        return data

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
