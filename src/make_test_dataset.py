import numpy as np
import tf_dataset as tfd

# 4 chunks
g_n_chunks = 4
# 10 numbers each
g_n_samples_per_chunk = 10
# 2 different sets of features
    # evens and odds
g_n_even_features = 2
g_n_odd_features = 3
# 1 label
    # label is the index

def make_even_features(global_index):
    feats = np.array(range(int(global_index*g_n_even_features), int((1+global_index)*g_n_even_features))) * 2
    return feats

def make_odd_features(global_index):
    feats = np.array(range(int(global_index*g_n_odd_features), int((1+global_index)*g_n_odd_features))) * 2 + 1
    return feats

def make_test_features():
    # first set of chunks
    for i in range(g_n_chunks):
        chunk_name = '.'.join([str(i), 'x'])
        even_features = np.zeros([g_n_samples_per_chunk, g_n_even_features])
        odd_features = np.zeros([g_n_samples_per_chunk, g_n_odd_features])
        labels = np.zeros([g_n_samples_per_chunk, 2]) # label + index

        for j in range(g_n_samples_per_chunk):
            global_index = i*g_n_samples_per_chunk + j
            even_features[j] = make_even_features(global_index)
            odd_features[j] = make_odd_features(global_index)
            labels[j] = np.array([global_index+1, global_index])

        np.savetxt('even.'+chunk_name+'.csv', even_features)
        np.save('even.'+chunk_name+'.npy', even_features)

        np.savetxt('odd.'+chunk_name+'.csv', odd_features)
        np.save('odd.'+chunk_name+'.npy', odd_features)

        np.savetxt('labels.'+str(i)+'.r', labels)

def test_tf_dataset4():
    # test all chunks with get_next_batch
    cg = tfd.ChunkGroup(pathFeatures=[['.'.join(['even', str(i), 'x.npy']) for i in range(1)],
                                ['.'.join(['odd', str(i), 'x.npy']) for i in range(1)]],
                    pathLabels=['.'.join(['labels', str(i), 'r']) for i in range(1)],
                    batch_size = 4)

    print 'batch_size', 4
    for i in range(5):
        even, odd, label, loop = cg.get_next_batch()
        print 'label', label
        for e, o, l in zip(even, odd, label):
            gen_evens = make_even_features(l)
            gen_odds = make_odd_features(l)

            assert all(gen_evens == e)
            assert all(gen_odds == o)

    # check keys for current_chunk
    print 'check keys are good'
    labels = cg.labels.current_set.allData
    keys = cg.labels.current_set.keys

    assert all(labels == keys-1)

def test_tf_dataset3():
    # test all chunks with get_next_batch
    cg = tfd.ChunkGroup(pathFeatures=[['.'.join(['even', str(i), 'x.npy']) for i in range(1)],
                                ['.'.join(['odd', str(i), 'x.npy']) for i in range(1)]],
                    pathLabels=['.'.join(['labels', str(i), 'r']) for i in range(1)],
                    batch_size = 4)

    loop = False
    print 'batch_size', 4
    while not loop:
        even, odd, label, loop = cg.get_onetime_batch()
        print 'label', label
        for e, o, l in zip(even, odd, label):
            gen_evens = make_even_features(l)
            gen_odds = make_odd_features(l)

            assert all(gen_evens == e)
            assert all(gen_odds == o)

    cg.reset()

    print 'batch_size', 4
    for i in range(5):
        even, odd, label, loop = cg.get_next_batch()
        print 'label', label
        for e, o, l in zip(even, odd, label):
            gen_evens = make_even_features(l)
            gen_odds = make_odd_features(l)

            assert all(gen_evens == e)
            assert all(gen_odds == o)

def test_tf_dataset2():
    # test all chunks with get_next_batch
    cg = tfd.ChunkGroup(pathFeatures=[['.'.join(['even', str(i), 'x.npy']) for i in range(g_n_chunks)],
                                ['.'.join(['odd', str(i), 'x.npy']) for i in range(g_n_chunks)]],
                    pathLabels=['.'.join(['labels', str(i), 'r']) for i in range(g_n_chunks)],
                    batch_size = 4)

    loop = False
    print 'batch_size', 4
    while not loop:
        even, odd, label, loop = cg.get_onetime_batch()
        print 'label', label
        for e, o, l in zip(even, odd, label):
            gen_evens = make_even_features(l)
            gen_odds = make_odd_features(l)

            assert all(gen_evens == e)
            assert all(gen_odds == o)

def test_tf_dataset1():
    # test all chunks with get_next_batch
    cg = tfd.ChunkGroup(pathFeatures=[['.'.join(['even', str(i), 'x.npy']) for i in range(g_n_chunks)],
                                ['.'.join(['odd', str(i), 'x.npy']) for i in range(g_n_chunks)]],
                    pathLabels=['.'.join(['labels', str(i), 'r']) for i in range(g_n_chunks)],
                    batch_size = 4)

    print 'batch_size', 4
    for epoch in range(2):
        print 'epoch', epoch
        loop = False
        while not loop:
            even, odd, label, loop = cg.get_next_batch()
            print 'label', label
            for e, o, l in zip(even, odd, label):
                gen_evens = make_even_features(l)
                gen_odds = make_odd_features(l)

                assert all(gen_evens == e)
                assert all(gen_odds == o)

        cg.reset()
        cg.randomize()

def test_tf_dataset5():
    # test all chunks with get_next_batch
    cg = tfd.ChunkGroup(pathFeatures=[['.'.join(['even', str(i), 'x.npy']) for i in range(g_n_chunks)],
                                ['.'.join(['odd', str(i), 'x.npy']) for i in range(g_n_chunks)]],
                    pathLabels=['.'.join(['labels', str(i), 'r']) for i in range(g_n_chunks)],
                    batch_size = 4)

    print cg.current_label_name()

def test_tf_dataset6():
    # test all chunks with get_next_batch
    cg = tfd.ChunkGroup(pathFeatures=[['.'.join(['even', str(i), 'x.npy']) for i in range(g_n_chunks)],
                                ['.'.join(['odd', str(i), 'x.npy']) for i in range(g_n_chunks)]],
                    pathLabels=['.'.join(['labels', str(i), 'r']) for i in range(g_n_chunks)],
                    batch_size = 4,
                    progress_filename='start_chunk.txt')

    print 'batch_size', 4
    for i in range(15):
        even, odd, label, loop = cg.get_next_batch()
        print 'label', label
        for e, o, l in zip(even, odd, label):
            gen_evens = make_even_features(l)
            gen_odds = make_odd_features(l)

            assert all(gen_evens == e)
            assert all(gen_odds == o)


if __name__ == '__main__':
    #make_test_features()
    #full_loop_test()

    #print("test1")
    #test_tf_dataset1()
    #print("test2")
    #test_tf_dataset2()
    #print("test3")
    #test_tf_dataset3()
    #print("test4")
    #test_tf_dataset4()
    #print("test5")
    #test_tf_dataset5()
    print("test6")
    test_tf_dataset6()
