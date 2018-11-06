from q2_extractor.Extractor import q2Extractor

class MetaHCRExtractor(q2Extractor):
    def extract_data(self):
        data = q2Extractor.extract_data(self)
        if self.type=='SampleData[DADA2Stats]':
            data.rename(columns={'sample-id': 'sample_name',
                                 'input':'lib_size_before_quality_control',
                                 'filtered': 'seq_quality_check',
                                 'merged': 'assembly',
                                 'non-chimeric': 'lib_size_after_control'}, inplace=True)
            data.drop(labels=['denoised'], axis=1, inplace=True)
            plugin_name = self.plugin.split(":")[-1]
            data['quality_check_tool'] = plugin_name
            data['quality_control_tool'] = plugin_name
            data['quality_lib_const_meth'] = plugin_name
            data['quality_lib_const_meth'] = self.plugin_versions[plugin_name]['version']
        elif self.type=='FeatureTable[Frequency]':
            data_normal = pd.DataFrame(normalize(data, norm='l1'))
            data_normal.rename(columns=dict(zip(data_normal.columns.values, 
                                                data.columns.values)), 
                               inplace=True)
            data_normal.rename(index=dict(zip(data_normal.index.values, data.index.values)),
                               inplace=True)
            data = data_normal.round(decimals=2)
            data = data.T
        elif self.type=='FeatureData[Taxonomy]':
            data.rename(columns={'Feature ID': 'organism_id'},
                       inplace=True)
            data[["kingdom","phylum","class",
                  "order","family","genus","species"]] = data['Taxon'].str.split("; ", expand=True)
            data.drop(labels=['Confidence', 'Taxon'], axis=1, inplace=True)
        #Skip if the q2Extractor result doesn't need to be mangled
        elif self.type in ['PCoAResults']:
            pass
        else:
            print("Warning: QIIME2 type '%s' not yet implemented, returning q2Extractor class results" % (self.type,))
        return data

def generate_SingleGeneAnalysis_table(taxonomy_artifact, table_artifact):
    merged_asv = table_artifact.extract_data().T.merge(taxonomy_artifact.extract_data(), 
                                                        left_index=True,
                                                        right_on='organism_id')
    merged_asv.rename(index=dict(zip(merged_asv.index.values, merged_asv['organism_id'])), inplace=True)
    merged_asv.drop(labels=["organism_id"], axis="columns", inplace=True)
    return merged_asv
