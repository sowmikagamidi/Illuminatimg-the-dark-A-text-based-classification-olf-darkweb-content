My project is about the Dark Web, known for ensuring user anonymity, has increasingly become a hub for illegal
activities and cyber threats, posing significant challenges for monitoring and regulation. This
study addresses these challenges by developing a robust classification model to identify Dark
Web services associated with malicious activities. Traditional methods like TF-IDF, Document
Matrix, and Latent Semantic Analysis often fail to exclude irrelevant data, limiting their accuracy.
To overcome these limitations, we propose a deep learning-based approach that integrates Topic
Modelling with a Text Convolutional Neural Network (TextCNN).
Our methodology begins with data collection from a Kaggle dataset, followed by preprocessing
to clean and standardize the text. Latent Dirichlet Allocation (LDA) is employed to extract 90
topic weights, which serve as enriched features for the TextCNN model. This combination
enhances the model's ability to capture contextual and thematic patterns in Dark Web content. We
compare the performance of our proposed LDA-TextCNN model against traditional algorithms
like K-Nearest Neighbors (KNN) and Random Forest, achieving a superior prediction accuracy
of 95%. Further, we introduce an extended LDA-Hybrid TextCNN model, which combines
features from TextCNN with a CNN2D architecture, incorporating dropout layers to reduce
overfitting. This hybrid model achieves an even higher accuracy of 96%, demonstrating its
effectiveness in classifying Dark Web services.
The results highlight the advantages of leveraging advanced deep learning techniques and topic
modelling for Dark Web classification. Our approach not only improves detection accuracy but
also provides a scalable and adaptable solution for real-world cybersecurity applications. Future
research aims to refine real-time classification capabilities and further integrate topic modelling
with deep learning to enhance threat detection in the dynamic Dark Web environment.


I have created a website related to that to detect the category of dataset taken.
