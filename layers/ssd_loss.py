import tensorflow as tf


class SSDLoss(object):
    def __init__(self, neg_pos_ratio=3, num_neg_min=0, alpha=1.0):
        self.neg_pos_ratio = neg_pos_ratio
        self.num_neg_min = num_neg_min
        self.alpha = alpha

    @staticmethod
    def smooth_l1_loss(y_true, y_pred):
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * tf.square(y_true - y_pred)
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    @staticmethod
    def log_loss(y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-15)
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.num_neg_min = tf.constant(self.num_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0]
        num_boxes = tf.shape(y_pred)[1]

        # compute all cls and loc loss
        classification_loss = tf.to_float(self.log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]))
        localization_loss = tf.to_float(self.smooth_l1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]))

        # cls loss for pos and neg
        negatives = y_true[:, :, 0]
        positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))
        n_positive = tf.reduce_sum(positives)
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)
        neg_class_loss_all = classification_loss * negatives
        num_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32)
        num_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.num_neg_min),
                                       num_neg_losses)

        def f1():
            return tf.zeros([batch_size])

        def f2():
            neg_class_loss_all_1d = tf.reshape(neg_class_loss_all, [-1])
            values, indices = tf.nn.top_k(neg_class_loss_all_1d,
                                          k=num_negative_keep,
                                          sorted=False)
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1d))
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, num_boxes]))
            neg_class_loss1 = tf.reduce_sum(classification_loss * negatives_keep,
                                            axis=-1)
            return neg_class_loss1

        neg_class_loss = tf.cond(tf.equal(num_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss

        # loc loss for positive
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)

        # total loss
        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive)
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss
