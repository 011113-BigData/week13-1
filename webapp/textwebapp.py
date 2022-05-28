# load the libraries
import datetime
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from waitress import serve
from flask import Flask, redirect, url_for, request, render_template
from pymongo import MongoClient
# suppress all warnings (ignore unnecessary warnings msgs)
import warnings
warnings.filterwarnings("ignore")

# define the flask and template directory 
app = Flask(__name__,template_folder='templates')

# initiate transformer
tfidf_transformer = TfidfTransformer()

# import the vocabulary
vocab_filename = "tfidf-vocab.model"
loaded_vocab = joblib.load(vocab_filename)

# import the model
model_filename = "MLP-tfidf.model"
loaded_model = joblib.load(model_filename)

# load the vector
loaded_vector = TfidfVectorizer(vocabulary=loaded_vocab)

# 127.0.0.1 is the local mongodb address installed
client = MongoClient('mongodb://127.0.0.1:27017/')

# YOU SHOULD change '<<yourUSERNAME>>' with userSTUDENTID (for example: user22222)
db = client['<<yourUSERNAME>>'] #<<yourUSERNAME>>

# serve the index 
@app.route("/")
def index():
    # retrieve last 5 data
    last_data = retrieve_lastdata(5)
    
    return render_template('form.html', last_data=last_data)

# handle the form action
@app.route("/result", methods=["POST"])
def prediction_result():
    
    # receiving the POST data from the client 
    # (Form submitted by the client/user)
    sentence = request.form.get('sentence')
    print('New sentence:', sentence)
    # convert / preprocess the new_sentence into tfidf vector
    input_vector = loaded_vector.fit_transform(np.array([sentence]))
    tfidf_input = tfidf_transformer.fit_transform(input_vector)

    # predict the label
    new_sentence_pred = loaded_model.predict(tfidf_input)

    if new_sentence_pred[0] == 0:
        output = "Negative"
    elif new_sentence_pred[0] == 1:
        output = "Neutral"
    elif new_sentence_pred[0] == 2:
        output = "Positive"
    else:
        output = "Undefined"
    
    print("Predicted as", output)
    
    # store the data into mongodb
    save_to_mongodb(sentence, output)
    
    return render_template('result.html', sentence=sentence, output=output)

# function to save the new sentence into mongodb
def save_to_mongodb(sentence, output):
    #text_prediction is the collection (table) name in our mongodb database
    text_prediction = db['text_prediction']

    # new record data, formatted in json
    new_record = { 
        'sentence': sentence, 
        'output': output, 
        'created_at': datetime.datetime.now()
    }

    # Insert one record
    text_prediction.insert_one(new_record)
    
# function to retrieve last data from mongodb
def retrieve_lastdata(limit):
    #text_prediction is the collection (table) name in our mongodb database
    text_prediction = db['text_prediction']

    # retrieve last 5 data (limit = 5)
    # sort by _id, -1 -> desc; 1 -> asc
    last_predictions_results = text_prediction.find().sort("_id", -1).limit(int(limit))

    #print the result:
    results = list()
    for data in last_predictions_results:
        created_at = data['created_at']
        date_time = created_at.strftime("%d/%m/%Y, %H:%M:%S")
        sentence = data['sentence']
        output = "Predicted as "+data['output']
        format_output = date_time+' - '+sentence+" ("+output+") "
        results.append(format_output)
        #print(format_output)
        
    return results

if __name__ == "__main__":
    '''
     # change the port number, available from 5100-5121 
     (there are 21 port slots, please choose one and post in the chat 
     so that other student can choose the available one)
    '''
    portNumber = 5101 # change this portnumber based on above slots
    hostAddress = '0.0.0.0' # public ip or change to '127.0.0.1' for localhost in your local computer
    print('The webapp can be accessed at', hostAddress+':'+str(portNumber))
    serve(app, host=hostAddress, port=portNumber)
    