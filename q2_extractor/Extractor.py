import zipfile
import yaml
import re
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
    regex = re.compile("[0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{12}")
    return regex.match(filename)[0]

class q2Extractor(object):
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
        yf = yaml.load(xf)
        self.type = yf['type']
        self.format = yf['format']
        
        #Next, hit up the action.yaml in the provenance folder
        #This is the provenance of THIS item
        xf = self.zfile.open(self.base_uuid + "/provenance/action/action.yaml")
        yf = yaml.load(xf)
        self.action_type = yf['action']['type']
        if (self.action_type == 'method') | (self.action_type == 'pipeline'):
            self.plugin = yf['action']['plugin']
            self.action = yf['action']['action']
            self.parameters = yf['action']['parameters']
            self.inputs = yf['action']['inputs']
            self.plugin_versions = yf['environment']['plugins']
        else:
            raise NotImplementedError("Action type '%s' not recognized." % (self.action_type,))
        self.env = yf['environment']
        
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
            prop_explained.rename(columns = dict(zip(prop_explained.columns.values, 
                                            principal_coords.index.values)),
                                  inplace = True)
            prop_explained.rename(index={0:"Proportion explained"}, inplace=True)
            eigvals.rename(columns = dict(zip(eigvals.columns.values, 
                                              principal_coords.index.values)),
                           inplace = True)
            eigvals.rename(index={0:"Eigenvalues"}, inplace=True)
            return {'eigenvalues': eigvals, 'proportion_explained': prop_explained, 
                    'coordinates': principal_coords}
        else:
            raise NotImplementedError("Type '%s' not yet implemented." % (self.type,))
            
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
