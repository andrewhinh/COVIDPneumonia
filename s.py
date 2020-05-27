def makeheatmap(file):
    import os
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow.keras import backend as K
    import keras
    import matplotlib.pyplot as plt
    from PIL import Image
    img_width, img_height = 28, 28
    model_path = './model/model.hdf5'
    from tensorflow.python.keras.saving.save import load_model
    model = load_model(model_path)

    from tensorflow.python.keras.backend import get_session

    session = get_session()
    init = tf.global_variables_initializer()
    session.run(init)

    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x).astype('float16')/255
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    class_output = model.output[:, answer]
    last_conv_layer = model.get_layer('conv2d_15')
    class_output = model.output[:, answer]
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(128):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap[x,y] = np.max(heatmap[x,y],0)

    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    plt.imshow(np.real(heatmap))

    x = load_img(file, target_size=(1024,1024))
    from skimage import transform
    upsample = transform.resize(heatmap, (1024,1024),preserve_range=True)
    plt.imshow(x)
    plt.imshow(upsample,alpha=0.4)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    plt.savefig('./uploads/' + file[8:-3] + 'new.png', bbox_inches='tight', pad_inches = 0)
    os.environ['TF2_BEHAVIOR'] = '1'
