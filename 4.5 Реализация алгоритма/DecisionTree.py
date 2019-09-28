import numpy as np
import random
class DecisionTree(): 
        def check_purity(self,data): 
            label_column = data[:, -1]
            unique_classes = np.unique(label_column)
            if len(unique_classes) == 1:
                return True
            else:
                return False
        def classify_data(self,data):
            label_column = data[:, -1]
            unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
            index = counts_unique_classes.argmax()
            classification = unique_classes[index]
            return classification
        def get_potential_splits(self,data):
            potential_splits = {}
            _,n_columns = data.shape
            for column_index in range(n_columns - 1):         
                values = data[:, column_index]
                unique_values = np.unique(values)
                potential_splits[column_index] = unique_values
            return potential_splits
        def split_data(self,data, split_column, split_value):
            split_column_values = data[:, split_column]
            type_of_feature = FEATURE_TYPES[split_column]
            if type_of_feature == "continuous":
                data_below = data[split_column_values <= split_value]
                data_above = data[split_column_values >  split_value]   
            else:
                data_below = data[split_column_values == split_value]
                data_above = data[split_column_values != split_value]
            return data_below, data_above
        def calculate_entropy(self,data):
            label_column = data[:, -1]
            _,counts = np.unique(label_column, return_counts=True)
            probabilities = counts / counts.sum()
            entropy = sum(probabilities * -np.log2(probabilities))
            return entropy
        def calculate_overall_entropy(self,data_below, data_above):
            n = len(data_below) + len(data_above)
            p_data_below = len(data_below) / n
            p_data_above = len(data_above) / n

            overall_entropy =  (p_data_below * self.calculate_entropy(data_below) 
                              + p_data_above * self.calculate_entropy(data_above))
            return overall_entropy
        def determine_best_split(self,data, potential_splits):
            overall_entropy = 9999
            for column_index in potential_splits:
                for value in potential_splits[column_index]:
                    data_below, data_above = self.split_data(data, split_column=column_index, split_value=value)
                    current_overall_entropy = self.calculate_overall_entropy(data_below, data_above)

                    if current_overall_entropy <= overall_entropy:
                        overall_entropy = current_overall_entropy
                        best_split_column = column_index
                        best_split_value = value

            return best_split_column, best_split_value
        def determine_type_of_feature(self,df):
            feature_types = []
            n_unique_values_treshold = 15
            for feature in df.columns:
                if feature != "label":
                    unique_values = df[feature].unique()
                    example_value = unique_values[0]

                    if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                        feature_types.append("categorical")
                    else:
                        feature_types.append("continuous")
            return feature_types
        def classify_example(self,example, tree):
            question = list(tree.keys())[0]
            feature_name, comparison_operator, value = question.split(" ")
            if comparison_operator == "<=":
                if example[feature_name] <= float(value):
                    answer = tree[question][0]
                else:
                    answer = tree[question][1]
            else:
                if str(example[feature_name]) == value:
                    answer = tree[question][0]
                else:
                    answer = tree[question][1]
            if not isinstance(answer, dict):
                return answer
            else:
                residual_tree = answer
                return self.classify_example(example, residual_tree)
        def predictedValues(self,df, tree):
            df["classification"] = df.apply(self.classify_example, args=(tree,), axis=1)
            return (df["classification"])
        
        def decision_tree_algorithm(self,df, counter=0, min_samples=2, max_depth=5): #Основной алгоритм работы
       
            if counter == 0:
                global COLUMN_HEADERS, FEATURE_TYPES
                COLUMN_HEADERS = df.columns
                FEATURE_TYPES = self.determine_type_of_feature(df)
                data = df.values
            else:
                data = df           
            if (self.check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
                classification = self.classify_data(data)
                return classification
            else:    
                counter += 1 
                potential_splits = self.get_potential_splits(data)
                split_column, split_value = self.determine_best_split(data, potential_splits)
                data_below, data_above = self.split_data(data, split_column, split_value)
                if len(data_below) == 0 or len(data_above) == 0:
                    classification = self.classify_data(data)
                    return classification
                feature_name = COLUMN_HEADERS[split_column]
                type_of_feature = FEATURE_TYPES[split_column]
                if type_of_feature == "continuous":
                    question = "{} <= {}".format(feature_name, split_value)
                else:
                    question = "{} = {}".format(feature_name, split_value)
                sub_tree = {question: []}
                yes_answer = self.decision_tree_algorithm(data_below, counter, min_samples, max_depth)
                no_answer = self.decision_tree_algorithm(data_above, counter, min_samples, max_depth)
                if yes_answer == no_answer:
                    sub_tree = yes_answer
                else:
                    sub_tree[question].append(yes_answer)
                    sub_tree[question].append(no_answer)
                return sub_tree