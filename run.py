from models.main_models.GM3H import HotelDataGenerator, LearnToPayAttention
import tensorflow as tf
from PIL import ImageFile
import pickle

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    BS = 16
    dg = HotelDataGenerator(BS)
    attention_model = LearnToPayAttention(bs=BS)
    attention_model.build_model()
    attention_model.model.summary()
    #attention_model.model.set_weights(pickle.load(open("hotel/gm3h_city_1ep_1e5.pkl", "rb")))
    attention_model.run_model(data_generator=dg, epochs=5)

    weigh = attention_model.model.get_weights()
    pklfile = "./../models/pretrained_geolocation/gm3h_new.pkl"
    try:
        fpkl = open(pklfile, 'wb')  # Python 3
        pickle.dump(weigh, fpkl, protocol=pickle.HIGHEST_PROTOCOL)
        fpkl.close()
    except:
        fpkl = open(pklfile, 'w')  # Python 2
        pickle.dump(weigh, fpkl, protocol=pickle.HIGHEST_PROTOCOL)
        fpkl.close()

    target_dict = dg.get_target()
    attention_model.test_model(target_dict)
