# Kalman Filter in tensorflow
# MIT Licence.
# Based on article published https://www.cs.utexas.edu/~teammco/misc/kalman_filter/
#
#  run: $ python3 kalman_filter.py
#


import tensorflow as tf
import numpy as np
import cv2

mp = np.array((2, 1), np.float32)              # measurement

##
# Note: Pass values to MP if not in DEBUG mode
##
if __debug__:
    meas=[]
    pred=[]
    frame = np.zeros((750, 500, 3), np.uint8)      # drawing canvas
    tp = np.zeros((2, 1), np.float32)              # tracked / prediction
    est = np.zeros((2, 1), np.float32)             # tracked / prediction

    def onmouse(k, x, y, s, p):
        global mp, meas
        mp = np.array([[np.float32(x)], [np.float32(y)]])
        meas.append((x, y))

    def paint():
        global frame, meas, pred
        for i in range(len(meas)-1): cv2.line(frame, meas[i], meas[i+1],(0, 0, 255))
        for i in range(len(pred)-1): cv2.line(frame, pred[i], pred[i+1],(255, 0, 0))

    def reset():
        global meas, pred, frame
        meas=[]
        pred=[]
        frame = np.zeros((400, 400, 3), np.uint8)

    cv2.namedWindow("kalman")
    cv2.setMouseCallback("kalman",onmouse);



class KalmanFilter:

    def __init__(self):
        pass

    # Const Params
    with tf.variable_scope("kf_constants"):
        F = tf.constant([
            [1, 0, 0.2, 0],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=tf.float32, name="kf_F")
        B = tf.constant([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=tf.float32, name="kf_B")
        H = tf.constant([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=tf.float32, name="kf_H")
        Q = tf.constant([
            [0.001, 0, 0, 0],
            [0, 0.001, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]], dtype=tf.float32, name="kf_Q")
        R = tf.constant([
            [0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1]], dtype=tf.float32, name="kf_R")

    # Inputs and Outputs
    with tf.variable_scope("kf_inputs_outputs"):
        x0 = tf.placeholder(dtype=tf.float32, shape=(4, 1), name="kf_x0") # Last coordinates
        P = tf.Variable([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]], dtype=tf.float32, name="kf_P") # 4 dynamic parameter: coordinates and velocity

    # Predict
    with tf.variable_scope("kf_predict"):
        xhat = tf.Variable([
            [0],
            [0],
            [0],
            [0]], dtype=tf.float32, name="kf_xhat")
        predict_xhat = tf.assign(xhat, tf.matmul(F, x0), name="kf_predict_xhat")
        predict_P = tf.assign(P, tf.matmul(F, tf.matmul(P, F, transpose_b=True)) + Q, name="kf_predict_P")

    # Correction
    with tf.variable_scope("kf_correction"):
        S = tf.matmul(H, tf.matmul(P, H, transpose_b=True)) + R
        K = tf.matmul(tf.matmul(P, H, transpose_b=True), tf.matrix_inverse(S))

        z = tf.matmul(H, x0, name="kf_z")
        y1 = z - tf.matmul(H, xhat)
        update_xhat = tf.assign_add(xhat, tf.matmul(K, y1), name="kf_update_xhat")
        delta_P = tf.matmul(K, tf.matmul(H, P))
        update_P = tf.assign_sub(P, delta_P, name="kf_update_P")
        init = tf.global_variables_initializer()


if __debug__:
    if __name__ == '__main__':
        while __debug__:
            with tf.Session() as sess:
                kf = KalmanFilter()
                sess.run(kf.init)
                ins = np.empty((4, 1), dtype=np.float32)

                ins[0] = mp[0]
                ins[1] = mp[1]
                est1, _, tp, _, _ = sess.run([kf.predict_xhat, kf.predict_P, kf.update_xhat, kf.update_P, kf.z],feed_dict={kf.x0: ins})


                pred.append((int(tp[0]), int(tp[1])))

                paint()
                cv2.imshow("kalman", frame)
                k = cv2.waitKey(30) &0xFF
                if k == 27: break
                if k == 32: reset()
