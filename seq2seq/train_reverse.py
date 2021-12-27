from from_book.data import *
from from_book.trainer import *
from models import Seq2seq
from optimizers import Adam

(x_train, t_train), (x_test, t_test) = load_data('addition.txt')
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
char_to_id, id_to_char = get_vocab()

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

rev_acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question,
                                    correct, id_to_char, verbose)

    acc = float(correct_num) / len(x_test)
    rev_acc_list.append(acc)
    print('검증 정확도 %.3f%%' % (acc * 100))
