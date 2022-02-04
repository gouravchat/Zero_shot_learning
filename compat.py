from numpy import matmul
import tensorflow as tf




def getProbsSoftmax(proj, labels):

    """ Softmax compatibility"""
    logits = tf.matmul(proj,labels,transpose_b = True)
    
    print("Logits: ",logits)

    probs = tf.nn.softmax(logits, axis=1)

    print("probs: ",probs)

    indexes = tf.argmax(probs, axis = 1)
    print("indexes: ",indexes)

    return indexes.numpy()



if __name__ == "__main__" :


    a = tf.random.uniform(shape=[5,4],dtype = tf.float32)
    b = tf.random.uniform(shape=[2,4],dtype = tf.float32)

    indexes = getProbsSoftmax(a, b)

    print(indexes)

    