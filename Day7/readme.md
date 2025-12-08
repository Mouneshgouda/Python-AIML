## Introduction¶
- Next Word Prediction (also called Language Modeling) is the task of predicting what word comes next. It is one of the fundamental tasks of NLP.

<img width="867" height="247" alt="image" src="https://github.com/user-attachments/assets/bb7e9335-b848-463b-876c-abff395b7006" />

## Application Language Modelling¶
- 1) Mobile keyboard text recommandation

<img width="1280" height="600" alt="image" src="https://github.com/user-attachments/assets/c1e06bf9-69d7-4716-8723-aee52251ed49" />

- 2) Whenever we search for something on any search engine, we get many suggestions and, as we type new words in it, we get better recommendations according to our         searching context. So, how will it happen???

  <img width="1014" height="629" alt="image" src="https://github.com/user-attachments/assets/9f959093-9048-436c-b46a-26c1eb4872a0" />

- It is poosible through natural language processing (NLP) technique. Here, we will use NLP and try to make a prediction model using Bidirectional LSTM (Long short-term -   memory) model that will predict next words of sentence.

- Titles text into sequences and make n_gram model¶
  suppose, we have sentence like "I am Yash" and this will convert into a sequence with their respective tokens {'I': 1,'am': 2,'Yash': 3}. Thus, output will be           [ '1'   ,'2' ,'3' ]

- Likewise, our all titles will be converted into sequences.

- Then, we will make a n_gram model for good prediction.

<img width="675" height="612" alt="image" src="https://github.com/user-attachments/assets/8da6fb9a-af12-41d7-ba42-3027fb72cbb4" />



Below image explain about everything.

