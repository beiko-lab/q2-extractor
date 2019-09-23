import os
from q2_extractor.Extractor import Extractor, get_default_args
import io
import numpy as np
import pandas as pd
import uuid
import copy
from qiime2.core.type.signature import __NoValueMeta as NoValueMeta

class DirectoryExtractor(object):
    """This class takes in a directory and outputs information on the operations
    that were performed within
    """

    def __init__(self, path="."):
        path = os.path.abspath(path)
        files = os.listdir(path)
        self.values_dict = {}
        aggregate_table = pd.DataFrame()
        for f in files:
            if f.endswith(".qza") | f.endswith(".qzv"):
                try:
                    ext = Extractor(path + "/" + f)
                except NotImplementedError:
                    print(f)
                    continue
                prov_table = ext.get_provenance(include_input=False)
                prov_table = prov_table.drop(["result_source", "result_type", "analysis_id", "analysis_date"], axis=1, errors='ignore')
                aggregate_table = pd.concat([aggregate_table, prov_table], sort=False).drop_duplicates()
        self.table = aggregate_table.reset_index().drop("result_id", axis=1).drop_duplicates()
        cols = list(self.table)
        pop_order = ["value_target", "value_type"] + \
                    sorted([x for x in cols if "upstream_step" in x], reverse=True) + \
                    ["step_id"]
        for item in pop_order:
            cols.insert(0, cols.pop(cols.index(item)))
        self.table = self.table.loc[:, cols]
        self.table = self.table.sort_values(["step_id", "upstream_step"])
        self.defaults = {}
    # This method gets a sheet that contains all Steps, their observed upstream 
    # relationships (traced via input files), descriptions, and defaults (scraped)
    # from QIIME2's sdk module
    def get_steps(self):
        for step in np.unique(self.table["step_id"]):
           if "__" in step: # This means the Extractor output it as a proper QIIME plugin, not an import or other function
               plugin_str = step.split("__")[0]
               func_str = step.split("__")[1].replace("-","_")
               default_args = get_default_args(plugin_str, func_str)
               if default_args:
                   self.defaults[step] = default_args
        p = self.table
        p = p.set_index("step_id")
        # Replace existing values in sheet with defaults
        # Grab each step's default parameters
        for step in self.defaults:
            for param in self.defaults[step]:
                if isinstance(self.defaults[step][param], NoValueMeta):
                    p.loc[step, param] = np.nan
                else:
                    p.loc[step, param] = self.defaults[step][param]
        # First cull: if the only difference was parameters, they get squashed
        p = p.drop_duplicates()
        # Find all steps that have multiple rows still (may be a valid reason?)
        dup = p.index[p.index.duplicated()]
        # So far, I've discovered that upstream_steps are often different after you replace the parameters with default
        # since different steps can feed into one another
        # so these loops squish those down so that they all have the same upstream_step values
        for step in dup:
            unique_ups = pd.Series(p.loc[step, [x for x in p.columns if "upstream_step" in x]].values.ravel('K')).dropna().unique()
            for idx, ups in enumerate(unique_ups):
                if idx == 0:
                    field = "upstream_step"
                else:
                    field = "upstream_step.%d" % (idx,)
                p.loc[step, field] = ups
        # Then we do another drop to see if that cleared it all up
        p = p.drop_duplicates()
        # To prepare for export, drop empty columns (no default)
        p = p.dropna(how='all', axis=1)
        # Move step_description to second column
        pop_order = ["step_description"]
        cols = p.columns.tolist()
        for item in pop_order:
            cols.insert(0, cols.pop(cols.index(item)))
        p = p.loc[:, cols]
        return p
        

    def get_process(self, name="New QIIME2 Pipeline", description=None):
        table = self.table
        table.loc[:,"process_category"] = "qiime2"
        table.loc[:,"process_id"] = name
        table.loc[:,"process_citation"] = "@article{bolyen2019reproducible,title={Reproducible, interactive, scalable and extensible microbiome data science using QIIME 2},author={Bolyen, Evan and Rideout, Jai Ram and Dillon, Matthew R and Bokulich, Nicholas A and Abnet, Christian C and Al-Ghalith, Gabriel A and Alexander, Harriet and Alm, Eric J and Arumugam, Manimozhiyan and Asnicar, Francesco and others},journal={Nature biotechnology},volume={37},number={8},pages={852--857},year={2019},publisher={Nature Publishing Group}}"
        table[:,"value_target.1"] = "process_id"
        table.columns = [x if "value_target" not in x else "value_target" for x in plugin_table.columns ]

        if description:
            table.loc[:,"process_description"] = description
        cols = list(table)
        if description:
            pop_order = ["process_citation", "process_description", "process_category", "process_id"]
        else:
            pop_order = ["process_citation", "process_category", "process_id"]
        for item in pop_order:
            cols.insert(0, cols.pop(cols.index(item)))
        table = table.loc[:, cols]
        return table

