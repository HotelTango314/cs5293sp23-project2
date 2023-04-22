import project2

def test_list_to_sent():
    text = ['rice','egg','soy sauce','shrimp']
    tester = project2.list_to_sent(text)
    assert tester == 'rice, egg, soy sauce, shrimp, '

def test_recommender():
    text = ['rice','egg','soy sauce','shrimp']
    N = 5
    test = project2.recommender(N,text)
    test1 = test['cuisine']=='chinese'
    test2 = len(test['closest'])==N
    assert test1 and test2
