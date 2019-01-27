\documentclass[a4paper,12pt]{article}
\usepackage{graphicx}
\begin{document}
    \title{
        \textbf{
            \underline{Computational Intelligence Report }
        }
    }
    \author{
        \textbf{
            Abhinav Kumar, Ayush Kumar Singh
        }
    }
    \date{Semester 1, M.Sc. Computer Science}
    \maketitle
    \section*{Dataset Introduction}
        \textbf{Name} : Hayes-Roth Data Set
        \newline\newline
        \textbf{Source} : UCI mahcine Learning Repository
        \newline\newline
        This is the link: \href{https://archive.ics.uci.edu/ml/datasets/Hayes-Roth}.
    \newline
    \section*{Attributes}
        Total \textbf{5} attributes
        \newline\newline
        1 class label
        \newline\newline
        4 attributes
    \newline
    \section*{Rows}
    There are total 132 instances
    \newline
    \section*{Neural Network}
    Layers in neural network in the project
    \newline
    \newline
    With 2 hidden layers
    \newpage
    \section*{Hyperparameters}
    1. First layer had 4 hidden units
    \newline\newline
    2. Second layer had 3 hidden units
    \section*{Activation functions used}
    1.Sigmoid : 
                $$
                \sigma(z) = \frac{1}{1 + e^{-(z)}}
                $$
    \newline
    2. tanh :  $$
                \frac{e^{z} + e^-{z}}{e^{z} - e^{-z}}
               $$
    3. ReLu(Rectified Linear Unit)
               $$
                \max{{0}, {Z}}
               $$
    \section*{On Breast Cancer Data Set}
    1. When weighted are built with inital value 0.005, initial weight decay.
    \newline\newline
    2. Learning rate = 0.2
    \newline\newline
    Accuracies and cost functions were as follows : 
    \newline\newline
    \textbf{Accuracy = 99\%, ReLu-Tanh}
    \newline\newline
    \graphicspath{ {./images/} }
    \includegraphics[scale=0.5]{bc_relu_tanh_99}
    \newline\newline\newline
    \textbf{Accuracy = 100\%, ReLu-ReLu}
    \newline\newline
    \graphicspath{ {./images/} }
    \includegraphics[scale=0.5]{bc_relu_relu_100.png}
    \newline
    \newline\newline\newline
    \textbf{Accuracy = 100\%, Tanh-Relu}
    \newline\newline
    \graphicspath{ {./images/} }
    \includegraphics[scale=0.5]{bc_tanh_relu_100}
    \newline
    \newline\newline\newpage
    \textbf{Accuracy = 100\%, Tanh-Tanh}
    \newline\newline\newlinw
    \graphicspath{ {./images/} }
    \includegraphics[scale=0.5]{bc_tanh_tanh_100.png}
    \newline
    
    
    
    
    
    
    
    
    
    
    \section*{On Hayes-Roth Data Set}
    1. When weighted are built with inital value 0.005, initial weight decay.
    \newline\newline
    2. Learning rate = 0.2
    \newline\newline
    Accuracies and cost functions were as follows : 
    \newline\newline
    \textbf{Accuracy = 85\%, ReLu-Tanh}
    \newline\newline
    \graphicspath{ {./images/} }
    \includegraphics[scale=0.5]{relu_tanh_85}
    \newline\newline\newline
    \textbf{Accuracy = 71\%, ReLu-ReLu}
    \newline\newline
    \graphicspath{ {./images/} }
    \includegraphics[scale=0.5]{relu_relu_71.png}
    \newline
    \newline\newline\newline
    \textbf{Accuracy = 87\%, Tanh-Relu}
    \newline\newline
    \graphicspath{ {./images/} }
    \includegraphics[scale=0.5]{tanh_relu_87}
    \newline
    \newline\newline\newpage
    \textbf{Accuracy = 90\%, Tanh-Tanh}
    \newline\newline\newlinw
    \graphicspath{ {./images/} }
    \includegraphics[scale=0.5]{tanh_tanh_90.png}
    \newline\newline
    \section*{Comparision between LogisticRegression an Neural network buily so far}
    \textbf{On Breast Cancer}
    \newline\newline
    \textbf{Accuracy: 95.95}\% By LogisticRegression 
    \newline\newline
    \textbf{Accuracy: 100}\% By Neural Netwrok 
    \newline
    \section*{Compairsion between Neural with 1 and 2 hidden layers}
    \textbf{Accuracy: 99}\% By 1 layer NN.
    \newline\newline
    \textbf{Accuracy: 100}\% By 2 layer NN.
\end{document}
