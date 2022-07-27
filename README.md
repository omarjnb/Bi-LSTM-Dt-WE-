# Bi-LSTM-Dt-WE-
An improved Bi-LSTM performance using domain-trained word embeddings for implicit aspect extraction


In aspect-based sentiment analysis ABSA, implicit aspects extraction is a fine-grained task aim for extracting the hidden aspect in the in-context meaning 
of the online reviews. Previous methods have shown that handcrafted rules interpolated in neural network architecture are a promising method for this task. 
In this work, we reduced the needs for the crafted rules that wastefully must be articulated for the new training domains or text data, instead proposing a 
new architecture relied on the multi-label neural learning. The key idea is to attain the semantic regularities of the explicit and implicit aspects using 
vectors of word embeddings and interpolate that as a front layer in the Bidirectional Long Short-Term Memory Bi-LSTM. First, we trained the proposed domain-trained 
word embeddings model, using explicit and implicit aspects. Second, using the trained word embeddings model as a front layer in the Bi-LSTM. Finally, extract implicit
aspects by testing the trained architecture using the opinionated reviews that comprise multiple implicit aspects. Our model outperforms several of the current 
methods for implicit aspect extraction.


