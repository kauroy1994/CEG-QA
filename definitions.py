#import statements
import logging
import pickle
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from copy import deepcopy

#define file I/O class
class FileIO(object):
    """
    contains functions for FileIO
    e.g., read and write
    functions are dataset specific
    """
    @staticmethod
    def read_pickle_file(data_path,datafile):
        """
        reads data files from data path
        """
        try:
            assert datafile in os.listdir(data_path) #make sure path contains the required pickle file
        except AssertionError as error:
            logging.error("No file called CLEVRER_data.pkl in data path\n"
                          "Exiting program ...")
            exit()
        file_path = os.path.join(data_path,datafile)
        with open(file_path,'rb') as fp:
            data = pickle.load(fp)
            return data
        
    @staticmethod
    def load_from_json():
        """
        loads trainer configuration from trainer_config.json file
        """
        try:
            #make sure trainer_config.json exists in current directory
            assert 'trainer_config.json' in os.listdir(os.curdir)
            with open('trainer_config.json') as f:
                return dict(json.load(f))
        except AssertionError as error:
            logging.error("trainer_config.json not found\n",
                          "Exiting with error %s",error)
            exit()

#define tokenizer class
class Tokenizer(object):
    """
    Implements methods for tokenizing strings,
    converting the tokens to numeric encodings,
    and decoding the numeric encodings back to tokens
    """

    def __init__(self):
        """
        class constructor stores tokenizer configuration
        reads tokens from tokens.json
        {
        'n_vocab': no. of tokens
        'tokens': token dictionary
        'idx': reverse token dictionary
        }
        """
        #read json file and load information
        token_file = open('tokens.json')
        self.tokens_without_UNK = dict(json.load(token_file))
        self.tokens = deepcopy(self.tokens_without_UNK)
        self.tokens.update({'UNK':len(self.tokens_without_UNK)})
        self.idx = dict()
        for token in self.tokens:
            self.idx[token] = self.tokens[token]
        self.n_vocab = len(self.tokens)

    def tokenize(self,input_string):
        """
        tokenizes input string and returns list of tokens
        """
        return [item if item in self.tokens_without_UNK else 'UNK' for item in input_string.split(' ')]
    
    def encode(self,input_string):
        """
        returns numeric encoding of input string tokens,
        after tokenization
        """
        return [self.tokens[item] for item in self.tokenize(input_string) if item in self.tokens]
    
    def decode(self,input_encoding):
        """
        converts numeric encodings back to tokens and returns
        """
        return [self.idx(encoding) for encoding in input_encoding if encoding in self.idx]

#define dataloader class
class Dataloader(object):
    """
    implements methods for:
    transforming the data for use by the model trainer
    """

    def __init__(self,context_size=None,tokenizer=None):
        """
        class constructor, for initializing data placeholder
        """
        self.data = list() #initialize place holder for data
        self.baseline = None #initialize flag to denote if baseline training process is invoked
        try:
            assert not tokenizer is None #make sure tokenizer is provided
            assert not context_size is None #make sure context size is provided
            self.tokenizer = tokenizer
            self.context_size = context_size
        except AssertionError as error:
            logging.error("Tokenizer not found or context size not set\n"
                          "Exiting program\n"
                          "error encountered %s",error)
            exit()

    def transform_point(self,data_point):
        """
        transforms data point for autoregressive generative modeling
        """
        question = data_point[0][0] #get question string
        answer = ' '.join(data_point[1]) #get answer string
        question_encoding = self.tokenizer.encode(question) #encode question
        answer_encoding = self.tokenizer.encode(answer) #encode answer
        #transform data point
        full_encoding = question_encoding + answer_encoding
        n_encoding = len(full_encoding)
        if self.baseline:
            baseline_embedding = data_point[0][1][self.baseline]
            return [[baseline_embedding,full_encoding[:i+1][-self.context_size:],full_encoding[i+1]] for i in range(n_encoding-1)]
        graph = [[self.tokenizer.encode(item[0]),self.tokenizer.encode(item[-1])] for item in data_point[0][1]['KG']] #extract knowledge graph
        graph_nodes = {}; graph_nodes_idx = {}; graph_edges = [] #initialize place holder for graph nodes and edges to return to trainer
        graph_node_counter = 0 #construct graph node index, and edge list in for loop below
        for edge in graph:
            for node in edge:
                if tuple(node) not in list(graph_nodes.values()):
                    graph_nodes[graph_node_counter] = tuple(node)
                    graph_nodes_idx[tuple(node)] = graph_node_counter; graph_node_counter += 1
            graph_edges.append([graph_nodes_idx[tuple(edge[0])],graph_nodes_idx[tuple(edge[1])]])
        return [[graph_nodes,graph_edges,full_encoding[:i+1][-self.context_size:],full_encoding[i+1]] for i in range(n_encoding-1)]
    
    def load_data(self,data,baseline):
        """
        loads data to the dataloader
        """
        self.baseline = baseline #set flag to denote if baseline training process is invoked
        for data_point in data: #transform and load each datapoint
            transformed_point = self.transform_point(data_point)
            self.data += transformed_point

#define the lagrangian computer class
class LagrangeMultipliers(nn.Module):
    """
    computes lagrange multipliers to
    balance the graph loss with token generation loss
    """

    def __init__(self,**config):
        """
        class constructor, sets up neural network config,
        input/output sizes, no. of layers, and hidden layer sizes
        """
        super().__init__() #call super class constructor
        emb_size = config["emb_size"] #get embedding size of input to network
        n_layers = config["n_layers"] #get no. of hidden layers for network
        h_size = config["h_size"] #network hidden layer size
        self.input_layer = nn.Linear(emb_size,h_size) #input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(h_size,h_size) for _ in range(n_layers)]) #hidden layers
        self.output_layer = nn.Linear(h_size,emb_size) #output layer
        def squash(x): 
            return F.relu(torch.sum(x))
        self.squash = squash #squash outputs to get scalar value for lagrange multiplier

    def forward(self,embeddings):
        """
        implements forward pass for the lagrange multiplier computation
        """
        activated_input_embeddings = F.leaky_relu(self.input_layer(embeddings)) #compute input layer embeddings
        activated_hidden_layer_embeddings = activated_input_embeddings #compute hidden layer embeddings using for loop
        for layer in self.hidden_layers:
            activated_hidden_layer_embeddings = F.leaky_relu(layer(activated_hidden_layer_embeddings))
        activated_output_layer_embeddings = F.leaky_relu(self.output_layer(activated_hidden_layer_embeddings)) #compute output layer embeddings
        return self.squash(activated_output_layer_embeddings[-1]) #squash and return scalar as lagrange multiplier

#define embedding integrator class
class EmbeddingsIntegrator(nn.Module):
    """
    implements methods to integrate two embeddings,
    of the same size
    """

    def __init__(self,**integrator_config):
        """
        class constructor, sets up integrator configuration,
        input/output sizes, no. of layers, and hidden layer sizes
        """
        super().__init__() #call super class constructor
        emb_size = integrator_config["emb_size"] #get embedding size of input to integrator network
        n_layers = integrator_config["n_layers"] #get no. of hidden layers for integrator network
        h_size = integrator_config["h_size"] #integrator network hidden layer size
        self.input_layer = nn.Linear(emb_size,h_size) #input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(h_size,h_size) for _ in range(n_layers)]) #hidden layers
        self.output_layer = nn.Linear(h_size,emb_size) #output layer

    def forward(self,embeddings):
        """
        implements the forward pass for the EmbeddingIntegrator network
        """
        activated_input_embeddings = F.leaky_relu(self.input_layer(embeddings)) #compute input layer embeddings
        activated_hidden_layer_embeddings = activated_input_embeddings #compute hidden layer embeddings using for loop
        for layer in self.hidden_layers:
            activated_hidden_layer_embeddings = F.leaky_relu(layer(activated_hidden_layer_embeddings))
        activated_output_layer_embeddings = F.leaky_relu(self.output_layer(activated_hidden_layer_embeddings)) #compute output layer embeddings
        return F.leaky_relu(torch.unsqueeze(torch.sum(activated_output_layer_embeddings,dim=0),0)) #return activated sum of embeddings

#define trainer class
class Trainer(nn.Module):
    """
    implements methods to train the model
    """

    def __init__(self,trainer_config):
        """
        class constructor, sets up model configuration, 
        tokenizer, dataloader, and parameters
        """
        super().__init__() #call super class constructor
        n_vocab = trainer_config["n_vocab"] #vocabulary size
        emb_size = trainer_config["emb_size"] #embedding size
        cx_size = trainer_config["cx_size"] #context size
        h_size = trainer_config["h_size"] #NN hidden layer size
        n_layers = trainer_config["n_layers"] #no. of NN hidden layers
        self.tokenizer = Tokenizer() #set tokenizer
        self.dataloader = Dataloader(context_size=cx_size,tokenizer=self.tokenizer) #set dataloader
        self.embeddings = nn.Embedding(n_vocab,emb_size) #embedding layer
        self.pos_embeddings = nn.Embedding(cx_size,emb_size) #position embedding layer
        self.projection_layer = nn.Linear(trainer_config["bg_emb_size"],emb_size) #baseline graph embedding projection layer
        self.integrator = EmbeddingsIntegrator(emb_size=emb_size,n_layers=n_layers,h_size=h_size) #embedding integrator network
        self.lagrange_multplier = LagrangeMultipliers(emb_size=emb_size,n_layers=n_layers,h_size=h_size) #lagrange multiplier network
        self.input_layer = nn.Linear(emb_size,h_size) #input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(h_size,h_size) for _ in range(n_layers)]) #hidden layers
        self.output_layer = nn.Linear(h_size,emb_size) #output layer
        self.classification_head = nn.Linear(emb_size,n_vocab) #classification head

    def load_training_data(self,data,baseline=None):
        """
        loads training data for the trainer
        """
        self.dataloader.load_data(data,baseline)

    def digraph_construct(self,graph_nodes):
        """
        constructs asymmetric graph from node embeddings
        """
        node_embeddings = [] #place holder for node embeddings
        for node in graph_nodes: #extract node embeddings
            node_embeddings.append(self.integrator(self.embeddings(torch.tensor(list(graph_nodes[node])))))
        node_embeddings = torch.row_stack(node_embeddings)
        eps = 1e03 #small epsilon value for log0 prevention
        n_embeddings = len(node_embeddings)
        approx_graph = [[0.0 for _ in range(n_embeddings)] for _ in range(n_embeddings)] #initialize approximate graph
        for i in range(n_embeddings): #compute approximate directed graph using bregman divergence in the for loop(s)
            embedding_i = node_embeddings[i]
            for j in range(n_embeddings):
                embedding_j = node_embeddings[j]
                approx_graph[i][j] = F.sigmoid(torch.sum(F.sigmoid(embedding_i) * torch.log(F.sigmoid(embedding_j)+eps)))
        return node_embeddings, approx_graph #return node_embeddings and approximate directed graph

    def forward(self,data_point):
        """
        implements forward pass for neural network training
        """
        if self.dataloader.baseline: #forward pass for baseline program
            graph_embedding = torch.tensor(data_point[0]) #extract graph embedding from baseline method
            context = torch.tensor(data_point[1]); context_embeddings = self.embeddings(context) #get context embedding
            position_embeddings = self.pos_embeddings(torch.arange(len(data_point[1]))) #get position embeddings
            projected_graph_embeddings = torch.row_stack([self.projection_layer(graph_embedding) for _ in range(context_embeddings.size()[0])])
            total_embeddings = context_embeddings+position_embeddings
            total_embeddings = self.integrator(torch.row_stack([total_embeddings,projected_graph_embeddings]))
            activated_input_embeddings = F.leaky_relu(self.input_layer(total_embeddings)) #compute input layer embeddings
            activated_hidden_layer_embeddings = activated_input_embeddings #compute hidden layer embeddings using for loop
            for layer in self.hidden_layers:
                activated_hidden_layer_embeddings = F.leaky_relu(layer(activated_hidden_layer_embeddings))
            activated_output_layer_embeddings = F.leaky_relu(self.output_layer(activated_hidden_layer_embeddings)) #compute output layer embeddings
            classification_head_output = F.leaky_relu(self.classification_head(activated_output_layer_embeddings)) #compute classification head
            logits = classification_head_output[-1] #extract last column as logits corresponding to n_vocab tokens
            return logits #return the logits
        
        if not self.dataloader.baseline: #forward pass for method described in the paper
            context = torch.tensor(data_point[-2]); context_embeddings = self.embeddings(context) #get context embedding
            position_embeddings = self.pos_embeddings(torch.arange(len(data_point[-2]))) #get position embeddings
            total_embeddings = context_embeddings+position_embeddings
            activated_input_embeddings = F.leaky_relu(self.input_layer(total_embeddings)) #compute input layer embeddings
            activated_hidden_layer_embeddings = activated_input_embeddings #compute hidden layer embeddings using for loop
            for layer in self.hidden_layers:
                activated_hidden_layer_embeddings = F.leaky_relu(layer(activated_hidden_layer_embeddings))
            activated_output_layer_embeddings = F.leaky_relu(self.output_layer(activated_hidden_layer_embeddings)) #compute output layer embeddings
            classification_head_output = F.leaky_relu(self.classification_head(activated_output_layer_embeddings)) #compute classification head
            logits = classification_head_output[-1] #extract last column as logits corresponding to n_vocab tokens
            graph_nodes = data_point[0]; n_nodes = len(graph_nodes)
            node_embeddings, approx_graph = self.digraph_construct(graph_nodes) #construct digraph and get node embeddings
            lagrange_multiplier = self.lagrange_multplier(node_embeddings)
            approx_graph = torch.stack([torch.stack(item) for item in approx_graph])
            return (logits,approx_graph,lagrange_multiplier)

    def train(self,epochs = 2):
        """
        implements code to train the model
        for different algorithms in the paper
        """
        optimizer = torch.optim.AdamW(self.parameters()) #set optimizer to Adam with weight decay
        loss_function = nn.CrossEntropyLoss() #set loss function to multi-label cross entropy loss
        if self.dataloader.baseline:
            for epoch in range(epochs):
                epoch_loss = 0.0
                for data_point in self.dataloader.data: #optimize over all datapoints
                    logits = self.forward(data_point)
                    targets = [0.0 for _ in range(self.dataloader.tokenizer.n_vocab)]; targets[data_point[-1]] = 1.0 #set target
                    targets = torch.tensor(targets) #convert target list to torch tensor
                    epoch_loss += loss_function(logits,targets) #compute loss function
                epoch_loss /= len(self.dataloader.data)
                print ("epoch_loss",epoch_loss.item()) #print loss to check if it is decreasing
                epoch_loss.backward(); optimizer.step(); optimizer.zero_grad() #perform gradient-based update
            return self #return trained model
        
        if not self.dataloader.baseline:
            for epoch in range(epochs):
                epoch_loss = 0.0
                for data_point in self.dataloader.data:
                    return_value = self.forward(data_point)
                    logits, approx_graph, lagrange_multiplier = return_value[0], return_value[1], return_value[2]
                    targets = [0.0 for _ in range(self.dataloader.tokenizer.n_vocab)]; targets[data_point[-1]] = 1.0 #set target
                    targets = torch.tensor(targets) #convert target list to torch tensor
                    loss1 = loss_function(logits,targets) #compute next token prediction loss function
                    graph_nodes = data_point[0]; n_nodes = len(graph_nodes) #get ground truth graph in next few lines of code
                    graph_edges = data_point[1]; gt_graph = [[0.0 for _ in range(n_nodes)] for _ in range(n_nodes)]
                    for edge in graph_edges:
                        gt_graph[edge[0]][edge[1]] = 1.0
                    gt_graph = torch.tensor(gt_graph)
                    loss2 = torch.sum(torch.pow(approx_graph-gt_graph,2)) #compute graph difference loss as sum of squared error as loss
                    epoch_loss += loss1 + lagrange_multiplier*loss2 #calculate constrained loss using lagrange multiplier
                epoch_loss /= len(self.dataloader.data)
                print ("epoch_loss",epoch_loss.item()) #print loss to check if it is decreasing
                epoch_loss.backward(); optimizer.step(); optimizer.zero_grad() #perform gradient-based update
            return self #return trained_model