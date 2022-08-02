# Chatbot-Project
## First attempt at creating a chatbot
Original Blog URL: https://data-flair.training/blogs/python-chatbot-project/

1.	*Intents.json* – The data file which has predefined patterns and responses.
2.	*train_chatbot.py* – In this Python file, we wrote a script to build the model and train our chatbot.
3.	*Words.pkl* – This is a pickle file in which we store the words Python object that contains a list of our vocabulary.
4.	*Classes.pkl* – The classes pickle file contains the list of categories.
5.	*Chatbot_model.h5* – This is the trained model that contains information about the model and has weights of the neurons.
6.	*Chatgui.py* – This is the Python script in which we implemented GUI for our chatbot. Users can easily interact with the bot.


# Planned: 

1. Probably can incoporate word-embeddings instead of just root word derived from `lemmatization`/`stemming`.

2. Experiment with a much more complex neural network than current 3-layered one.

3. See if we can monitor some metric of accuracy for responses: possibly add a thumbs-up / thumbs-down button for capturing user-satisfaction.