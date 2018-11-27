import os

from Extractor import q2Extractor

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

    def print_pipeline(self):
        print("name\tinput_types\toutput_types\tparameters")
        for action in self.actions:
            for plugin in self.actions[action]: 
                parameters = self.actions[action][plugin]
                param_str = ",".join([[":".join([x,str(d[x])]) for x in d ][0] \
                                          for d in parameters ])
                plugin = plugin.split(":")[-1]
                name = plugin + "__" + action
                input_str = ",".join([
                    self.action_inputs[action][in_name][0]['from'] \
                            for in_name in self.action_inputs[action]
                    ])
                output_str = ",".join([x["to"] \
                           for x in self.action_output[action]])
                print("%s\t%s\t%s\t%s" % (name,
                                          input_str,
                                          output_str,
                                          param_str))

            

