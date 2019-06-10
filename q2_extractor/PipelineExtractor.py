import os

from q2_extractor.Extractor import q2Extractor

class DirectoryInspector(object):
    """This class takes in a directory and outputs information on the operations
    that were performed within
    """

    def __init__(self, path="."):
        path = os.path.abspath(path)
        files = os.listdir(path)
        self.actions = {}
        self.action_inputs = {}
        self.action_output = {}
        for f in files:
            if f.endswith(".qza") | f.endswith(".qzv"):
                try:
                    ext = q2Extractor(path + "/" + f)
                except NotImplementedError:
                    print(f)
                    continue
                if ext.action not in self.actions:
                    self.actions[ext.action] = {ext.plugin: ext.parameters}
                    self.action_output[ext.action] = ext.output
                    self.action_inputs[ext.action] = ext.inputs
                else:
                    self.action_output[ext.action].extend(ext.output)
                    self.action_inputs[ext.action].update(ext.inputs)

    def get_pipeline(self, name="New QIIME2 Pipeline"):
        out_str = ""
        out_str += "pipeline_id\tpipeline_step_id\tpipeline_step_action\tpipeline_step_parameter_id\tpipeline_step_parameter_value\n"
        for action in self.actions:
            for plugin in self.actions[action]:
                plugin_name = plugin.split(":")[-1]
                parameters = []
                parameters = self.actions[action][plugin]
                input_str = ",".join([
                    self.action_inputs[action][in_name][0]['from'] \
                            for in_name in self.action_inputs[action]
                    ])
                output_str = ",".join([x["to"] \
                           for x in self.action_output[action]])
                if len(parameters) == 0:
                    out_str += "%s\t%s\t%s\t\t\n" % (name,
                                                     plugin_name,
                                                     action)
                for parameter in parameters:
                    for key, value in parameter.items():
                        param_str = ""
                        out_str += "%s\t%s\t%s\t%s\t%s\n" % (name,
                                                             plugin_name, 
                                                             action,
                                                             key,
                                                             value)
        return out_str

            

