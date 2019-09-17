import os

from q2_extractor.Extractor import q2Extractor
import io
import numpy as np
import pandas as pd
import uuid
import copy

class DirectoryInspector(object):
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
                    ext = q2Extractor(path + "/" + f)
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

    def get_process(self, name="New QIIME2 Pipeline", description=None):
        table = self.table
        table.loc[:,"process_category"] = "qiime2"
        table.loc[:,"process_id"] = name
        table.loc[:,"process_citation"] = "@article{bolyen2019reproducible,title={Reproducible, interactive, scalable and extensible microbiome data science using QIIME 2},author={Bolyen, Evan and Rideout, Jai Ram and Dillon, Matthew R and Bokulich, Nicholas A and Abnet, Christian C and Al-Ghalith, Gabriel A and Alexander, Harriet and Alm, Eric J and Arumugam, Manimozhiyan and Asnicar, Francesco and others},journal={Nature biotechnology},volume={37},number={8},pages={852--857},year={2019},publisher={Nature Publishing Group}}"
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

