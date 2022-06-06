Research in natural language processing has seen significant growth over the past decade due to successful application of parametric models such as deep neural networks.1 However, parametric models are dependent on access to large datasets and have difficulty generalizing from a small number of samples. These data-size limitations have led researchers to explore non-parametric models, such as k-nearest neighbors (k-NN), which are better at generalizing with a small number of examples.2 A subset of the research on non-parametric models evaluates the effectiveness of n-shot learning in classification problems, in which networks are trained on n labeled examples of each possible class. Drawing from papers that have investigated uses of n-shot learning in natural language processing, we will construct a k-nearest neighbors implementation that can predict sentiment of financial phrases.3 

