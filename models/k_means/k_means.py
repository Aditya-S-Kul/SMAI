import numpy as np
import random

class K_means:
    def __init__(self, k):
        self.k = k
        self.means = [0]*self.k

        

    def fit(self, X):
        n = len(X)
        d = len(X[0])

        self.C = [random.randint(0, self.k - 1) for _ in range(n)]
        self.C = np.array(self.C)
        cost = float('inf')

        # Randomly initializing k means

        # mistake in initializing means(initial means should be randomly picked from data samples but you didn't)
        # for i in range(self.k):
        #     self.means[i] = np.random.uniform(0, 600, d)

        unique_values = np.random.choice(n, size=self.k, replace=False)
        for i in range(self.k):
            self.means[i] = X[unique_values[i]]
        
        # self.means[0] = X[random.randint(0, int(n/3))]
        # self.means[1] = X[random.randint(int(n/3)+1, int(2*n/3)-1)]
        # self.means[2] = X[random.randint(int(2*n/3), n)]

        
        

        while True:
            # Checking convergence
            if abs(cost - self.getCost(X)) < 0.000001:
                break

            cost = self.getCost(X)


            # Assigning nearest means

            # for i in range(n):
            #     min = float('inf')
            #     for j in range(self.k):
            #         dist = np.dot(X[i] - self.means[j], X[i] - self.means[j])
                    
            #         if dist < min:
            #             min = dist
            #             self.C[i] = j


            # vetorized code for above
            distances = np.linalg.norm(X[:, np.newaxis] - self.means, axis=2) ** 2
            self.C = np.argmin(distances, axis=1)  # Assign each point to the closest mean

            
            # Updating means

            # for j in range(self.k):
            #     d = len(X[0])
            #     x = np.zeros(d)
            #     count = 0
            #     for i in range(n):
            #         if self.C[i] == j:
            #             x = x + X[i]
            #             count += 1
            #     if count != 0:
            #         self.means[j] = x/count
            
            # vetorized code for above
            self.means = np.array([X[self.C == j].mean(axis=0) if np.any(self.C == j) else self.means[j] for j in range(self.k)]) 


                        
    def predict(self):
        return self.C
    
    def getCost(self, X):

        # sum = 0
        # for i in range(len(X)):
        #     sum = sum + np.dot(X[i] - self.means[self.C[i]], X[i] - self.means[self.C[i]])
        # cost = sum
        
        # Ensure self.means is a NumPy array
        self.means = np.array(self.means)

        # vetorized code for above
        cost = np.sum(np.linalg.norm(X - self.means[self.C], axis=1) ** 2)

        return cost
    
    def getMeans(self):
        return self.means

