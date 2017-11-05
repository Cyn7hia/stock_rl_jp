from deeplearning_assistant.model_builder import AbstractModelBuilder


class MarketPolicyGradientModelBuilder(AbstractModelBuilder):
    def buildModel(self):
        from keras.models import Model
        from keras.layers import Conv2D, Input, Dense, Flatten, merge
        from keras.layers.advanced_activations import LeakyReLU

        B = Input(shape=(3,))
        b = Dense(5, activation="relu")(B)

        inputs = [B]
        merges = [b]

        for i in range(1):
            S = Input(shape=[2, 60, 1])
            inputs.append(S)

            h = Conv2D(2048, (1, 3))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(2048, (1, 5))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(2048, (1, 10))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(2048, (1, 20))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(2048, (1, 40))(S)
            h = LeakyReLU(0.001)(h)

            h = Flatten()(h)
            h = Dense(512)(h)
            h = LeakyReLU(0.001)(h)
            merges.append(h)

            h = Conv2D(2048, (1, 60))(S)
            h = LeakyReLU(0.001)(h)

            h = Flatten()(h)
            h = Dense(512)(h)
            h = LeakyReLU(0.001)(h)
            merges.append(h)

        m = merge(merges, mode='concat', concat_axis=1)
        m = Dense(1024)(m)
        m = LeakyReLU(0.001)(m)
        m = Dense(512)(m)
        m = LeakyReLU(0.001)(m)
        m = Dense(256)(m)
        m = LeakyReLU(0.001)(m)
        V = Dense(2, activation='softmax')(m)
        model = Model(input=inputs, output=V)

        print(model.summary())

        return model


class MarketModelBuilder(AbstractModelBuilder):
    def buildModel(self):
        # ノートPCのGPUではOOMになってしまうのでモデルを縮小している
        from keras.models import Model
        from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, merge
        from keras.layers.advanced_activations import LeakyReLU

        dr_rate = 0.0

        # 購買余力、ポジションサイズ、ポジション
        B = Input(shape=(3,))
        b = Dense(5, activation="relu")(B)

        inputs = [B]
        merges = [b]

        for i in range(1):
            S = Input(shape=[2, 60, 1])
            inputs.append(S)

            h = Conv2D(64, (1, 3))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(128, (1, 5))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(256, (1, 10))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(512, (1, 20))(S)
            h = LeakyReLU(0.001)(h)

            h = Flatten()(h)
            h = Dense(1024)(h)
            h = LeakyReLU(0.001)(h)
            h = Dropout(dr_rate)(h)
            merges.append(h)

            h = Conv2D(1024, (1, 60))(S)
            h = LeakyReLU(0.001)(h)

            h = Flatten()(h)
            h = Dense(2048)(h)
            h = LeakyReLU(0.001)(h)
            h = Dropout(dr_rate)(h)
            merges.append(h)

        m = merge(merges, mode='concat', concat_axis=1)
        m = Dense(1024)(m)
        m = LeakyReLU(0.001)(m)
        m = Dropout(dr_rate)(m)
        m = Dense(512)(m)
        m = LeakyReLU(0.001)(m)
        m = Dropout(dr_rate)(m)
        m = Dense(256)(m)
        m = LeakyReLU(0.001)(m)
        m = Dropout(dr_rate)(m)
        V = Dense(2, activation='linear', init='zero')(m)
        model = Model(input=inputs, output=V)

        print(model.summary())
        return model

    def buildModelOriginal(self):
        from keras.models import Model
        from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, merge
        from keras.layers.advanced_activations import LeakyReLU

        dr_rate = 0.0

        # 購買余力、ポジションサイズ、ポジション
        B = Input(shape=(3,))
        b = Dense(5, activation="relu")(B)

        inputs = [B]
        merges = [b]

        for i in range(1):
            S = Input(shape=[2, 60, 1])
            inputs.append(S)

            h = Conv2D(64, (1, 3))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(128, (1, 5))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(256, (1, 10))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(512, (1, 20))(S)
            h = LeakyReLU(0.001)(h)
            h = Conv2D(1024, (1, 40))(S)
            h = LeakyReLU(0.001)(h)

            h = Flatten()(h)
            h = Dense(2048)(h)
            h = LeakyReLU(0.001)(h)
            h = Dropout(dr_rate)(h)
            merges.append(h)

            h = Conv2D(2048, (1, 60))(S)
            h = LeakyReLU(0.001)(h)

            h = Flatten()(h)
            h = Dense(4096)(h)
            h = LeakyReLU(0.001)(h)
            h = Dropout(dr_rate)(h)
            merges.append(h)

        m = merge(merges, mode='concat', concat_axis=1)
        m = Dense(1024)(m)
        m = LeakyReLU(0.001)(m)
        m = Dropout(dr_rate)(m)
        m = Dense(512)(m)
        m = LeakyReLU(0.001)(m)
        m = Dropout(dr_rate)(m)
        m = Dense(256)(m)
        m = LeakyReLU(0.001)(m)
        m = Dropout(dr_rate)(m)
        V = Dense(2, activation='linear', init='zero')(m)
        model = Model(input=inputs, output=V)

        print(model.summary())

        return model
