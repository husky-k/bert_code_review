import tensorflow as tf

t1 = tf.random.uniform(shape=(4, 3, 1), minval=0, maxval=10, dtype=tf.int32)
t2 = tf.ones(shape=(4, 1, 3), dtype=tf.int32)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(t1))
print(sess.run(t2))
print(sess.run(t1 * t2))

sess.close()
