import time
#import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import load_model


from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os

tampon = os.path.join(os.getcwd(),"imagesEntrees","datas")

app=Flask(__name__)

# API avec JSON
@app.route('/reperage', methods=['POST'])
def get_reperage():
    raw_images = []
    images = request.files.getlist("img")
    image_names = []

    for image in images:
        image_name = image.filename
        image_names.append(image_name)
        image.save(os.path.join(tampon, image_name))
        image.save(os.path.join(os.getcwd(),"data","chargees", image_name))
        img_raw = tf.image.decode_image(
            open(tampon+"\\"+image_name, 'rb').read(), channels=3)
        raw_images.append(img_raw)
    
    response = []
    test_datagen = ImageDataGenerator(rescale =1./255)
    image_in = test_datagen.flow_from_directory(os.getcwd()+"\imagesEntrees\\", target_size=(64,64), color_mode='rgb', class_mode='categorical', batch_size=1, shuffle=False)

    for j in range(len(raw_images)):
        # Reponse en texte
        responses = []
        #raw_img = raw_images[j]

        t1 = time.time()
        #Prédiction
        modele= load_model('./model/MonModel.h5')
        predictions = modele.predict(image_in)

        t2 = time.time()
        print('time: {}'.format(t2 - t1))

    
        if predictions[0] < 0.5:
            predire = 'horse' 
        else :
            predire = 'cow'
        #valeur=predictions[0][np.argmax(predictions[0])]
        valeur = predictions[0]

    print('Repérages:')
    print('JE SUIS {}'.format(predictions[0]))
    
    responses.append({
        "classe": predire ,
        "confiance": "{}".format(valeur)})
    response.append({
        "image": image_names[0],
        "detections": responses})

    #Suppression de l'image intermédiaire
    # os.remove(tampon+"\\"+image_name)

    try:
        return jsonify({"reponse":response}), 200
    except FileNotFoundError:
        abort(404)
        
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=6000)