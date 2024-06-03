import numpy as np
import warnings

import pandas as pd


class Topsis():
    evaluation_matrix = np.array([])  # Matrix
    weighted_normalized = np.array([])  # Weight matrix
    normalized_decision = np.array([])  # Normalisation matrix
    M = 0  # Number of rows
    N = 0  # Number of columns

    '''
	Create an evaluation matrix consisting of m alternatives and n criteria,
	with the intersection of each alternative and criteria given as {\displaystyle x_{ij}}x_{ij},
	we therefore have a matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}}.
	'''

    def __init__(self, evaluation_matrix, weight_matrix, criteria):
        # M×N matrix
        self.evaluation_matrix = np.array(evaluation_matrix, dtype="float")

        # M alternatives (options)
        self.row_size = len(self.evaluation_matrix)

        # N attributes/criteria
        self.column_size = len(self.evaluation_matrix[0])

        # N size weight matrix
        self.weight_matrix = np.array(weight_matrix, dtype="float")
        self.weight_matrix = self.weight_matrix / sum(self.weight_matrix)
        self.criteria = np.array(criteria, dtype="float")

    '''
	# Step 2
	The matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}} is then normalised to form the matrix
	'''

    def step_2(self):
        # normalized scores
        self.normalized_decision = np.copy(self.evaluation_matrix)
        sqrd_sum = np.zeros(self.column_size)
        for i in range(self.row_size):
            for j in range(self.column_size):
                sqrd_sum[j] += self.evaluation_matrix[i, j] ** 2
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.normalized_decision[i,j] = self.evaluation_matrix[i, j] / (sqrd_sum[j] ** 0.5)

    '''
	# Step 3
	Calculate the weighted normalised decision matrix
	'''

    def step_3(self):
        from pdb import set_trace
        self.weighted_normalized = np.copy(self.normalized_decision)
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.weighted_normalized[i, j] *= self.weight_matrix[j]

    '''
	# Step 4
	Determine the worst alternative {\displaystyle (A_{w})}(A_{w}) and the best alternative {\displaystyle (A_{b})}(A_{b}):
	'''

    def step_4(self):
        self.worst_alternatives = np.zeros(self.column_size)
        self.best_alternatives = np.zeros(self.column_size)
        for i in range(self.column_size):
            if self.criteria[i]:
                self.worst_alternatives[i] = min(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = max(self.weighted_normalized[:, i])
            else:
                self.worst_alternatives[i] = max(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = min(self.weighted_normalized[:, i])

    '''
	# Step 5
	Calculate the L2-distance between the target alternative {\displaystyle i}i and the worst condition {\displaystyle A_{w}}A_{w}
	{\displaystyle d_{iw}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{wj})^{2}}},\quad i=1,2,\ldots ,m,}
	and the distance between the alternative {\displaystyle i}i and the best condition {\displaystyle A_{b}}A_b
	{\displaystyle d_{ib}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{bj})^{2}}},\quad i=1,2,\ldots ,m}
	where {\displaystyle d_{iw}}d_{{iw}} and {\displaystyle d_{ib}}d_{{ib}} are L2-norm distances 
	from the target alternative {\displaystyle i}i to the worst and best conditions, respectively.
	'''

    def step_5(self):
        self.worst_distance = np.zeros(self.row_size)
        self.best_distance = np.zeros(self.row_size)

        self.worst_distance_mat = np.copy(self.weighted_normalized)
        self.best_distance_mat = np.copy(self.weighted_normalized)

        for i in range(self.row_size):
            for j in range(self.column_size):
                self.worst_distance_mat[i][j] = (self.weighted_normalized[i][j] - self.worst_alternatives[j]) ** 2
                self.best_distance_mat[i][j] = (self.weighted_normalized[i][j] - self.best_alternatives[j]) ** 2

                self.worst_distance[i] += self.worst_distance_mat[i][j]
                self.best_distance[i] += self.best_distance_mat[i][j]

        for i in range(self.row_size):
            self.worst_distance[i] = self.worst_distance[i] ** 0.5
            self.best_distance[i] = self.best_distance[i] ** 0.5

    '''
	# Step 6
	Calculate the similarity
	'''

    def step_6(self):
        np.seterr(all='ignore')
        self.worst_similarity = np.zeros(self.row_size)
        self.best_similarity = np.zeros(self.row_size)

        for i in range(self.row_size):
            # calculate the similarity to the worst condition
            self.worst_similarity[i] = self.worst_distance[i] / \
                                       (self.worst_distance[i] + self.best_distance[i])

            # calculate the similarity to the best condition
            self.best_similarity[i] = self.best_distance[i] / \
                                      (self.worst_distance[i] + self.best_distance[i])

    def ranking(self, data):
        return [i  for i in data.argsort()]

    def rank_to_worst_similarity(self):
        # return rankdata(self.worst_similarity, method="min").astype(int)
        return self.ranking(self.worst_similarity)

    def rank_to_best_similarity(self):
        # return rankdata(self.best_similarity, method='min').astype(int)
        return self.ranking(self.best_similarity)

    def calc(self):
        # print("Step 1\n", self.evaluation_matrix, end="\n\n")
        self.step_2()
        # print("Step 2\n", self.normalized_decision, end="\n\n")
        self.step_3()
        # print("Step 3\n", self.weighted_normalized, end="\n\n")
        self.step_4()
        # print("Step 4\n", self.worst_alternatives,
        #       self.best_alternatives, end="\n\n")
        self.step_5()
        # print("Step 5\n", self.worst_distance, self.best_distance, end="\n\n")
        self.step_6()
        print(min(self.rank_to_best_similarity()))
        return self.rank_to_best_similarity()[:int(len(self.rank_to_best_similarity())*0.1)]
        # print("Step 6\n", self.worst_similarity,
        #       self.best_similarity, end="\n\n")
if __name__ == "__main__":

    paths = ["BuildingModels/Data/Facebook.csv"]
    for path in paths:
        df = pd.read_csv(path)
        # df = df.iloc[:, :]

        node = df.iloc[:,0].values
        degree = df['degree'].values
        closeness = df['closeness'].values
        betwennes = df['betwennes'].values
        evaluation_matrix = np.array([0,0,0])
        for d, c, b in zip(degree, closeness, betwennes):
            included = [d, c, b]
            included = np.array(included)
            evaluation_matrix = np.vstack((evaluation_matrix, included))
        evaluation_matrix = evaluation_matrix[1:]
        weights = [1, 1, 1]

        # print(df.iloc[1, :])
        '''
        if higher value is preferred - True
        if lower value is preferred - False
        '''
        criterias = np.array([True, True, True])

        t = Topsis(evaluation_matrix, weights, criterias)
        # print(t.rank_to_best_similarity)
        best = t.calc()
        # print(best)
        # print(b)
        # for d,b, c, be in zip(degree, best,closeness,betwennes):
        #     print(b, d, c, be)
        # df.loc[:,'influence'] = 0
        for b in best:
            # print(b,node[b])
            df.loc[df['Node']==node[b], 'influence'] = 1
        

        df.to_csv(path, index=False)
    # print(df.iloc[:, :])

    #print(t.best_similarity[55])
    #print(df.iloc[73, -1])
    # print(max(best))

    # # ranking = t.rank_to_best_similarity()
    #df['influence'] = best
    # # print(df.head())
    # df.to_csv("BuildingModels/Data_t.csv", index=False)
    # df = pd.read_csv("BuildingModels/Data/Science.csv")
    # dft = pd.read_csv("BuildingModels/Data/Football.csv")
    # dfm = pd.read_csv("BuildingModels/Data/Dolphins.csv")
    # dfs = pd.read_csv("BuildingModels/Data/Karate.csv")


    # concat = pd.concat([df, dft,dfm, dfs], )
    # # print(concat.head(), len(concat))

    # concat.to_csv("BuildingModels/Data/AllData.csv", index=False)
    # # print(t.best_similarity[36], t.best_similarity[36])

    # import networkx as nx # type: ignore

    # G = nx.Graph()
    # G.add_node('a')
    # G.add_node('b')
    # G.add_node('c')

    # G.add_edge('a','b')
    # G.add_edge('b','c')

    # degree = nx.degree_centrality(G)
    # closeness = nx.closeness_centrality(G)
    # betweenes = nx.betweenness_centrality(G)

    # evaluation_matrix = np.array([0,0,0])
    # for d, c, b in zip(degree.values(), closeness.values(), betweenes.values()):
    #     included = [d, c, b]
    #     included = np.array(included)
    #     evaluation_matrix = np.vstack((evaluation_matrix, included))
    # evaluation_matrix = evaluation_matrix[1:]
    # weights = [1, 1, 1]

    # '''
    # if higher value is preferred - True
    # if lower value is preferred - False
    # '''
    # criterias = np.array([True, True, True])

    # t = Topsis(evaluation_matrix, weights, criterias)

    # best = t.calc()
    # print(best)











