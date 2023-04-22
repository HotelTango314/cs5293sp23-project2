# PROJECT 2 README

## HENRY THOMAS [[[]]] CS5293 SP 23

### 1. HOW TO INSTALL

The project can be installed by running the following code:

pipenv install project2

---

### 2. HOW TO RUN

To run from the root directory of the pipenvironment it is recommended that th efollowing command be used to run the redactor tool:

```
pipenv run python project2.py --N 5 --ingredient 'soy sauce' --ingredient rice --ingredient shrimp --ingredient egg
```

project2.py is the name of the source file that performs the clustering of the yummly.json data as well as takes the new data introduced from the command line and produces the cuisine label, similar dishes, and associated scores.

--N is the number of similar dishes that the user would like the program to recommend. It is assumed that the choice of N will never exceed the number of recipes in the cluster.

--ingredient is the tag that precedes each ingredient in a dish for which the user is attempting to find similar dishes. Any number of ingredients can be included so long as the number of ingredients does not exceed the largest positive integer supported by Python which according to a call to sys.maxsize on my machine is 9223372036854775807. If a single ingredient is represented by two words, eg. Soy Sauce, the user should enclose this ingredient in quotes i.e. --ingredient "soy sauce".

Visit this link for a video example of how to run:
[youtube.com](url)

### 3. DESIGN CONSIDERATIONS

The system works by first importing a .json file containing yummly.com recipe data, specifically the type of cuisine and the list of ingredients, with an assigned ID included with each recipe. The ingredients lists, cuisine labels, and ID numbers are then extracted into 3 python lists, one for each feature.

The list of ingredients are to be viewed as documents and the problem we are trying to solve here reduces to document clustering. The list of ingredients starts out as a list of list, but the list of lists is converted into a list of strings, where each list of ingredients pulled from the yummly.com json file is turned into a single string i.e. [ingredient1, ingredient2, ingredient3] -> "ingredient1, ingredient2, ingredient3, ". At this point it is easy to see how each list of ingredients is its own document and all are to be clustered together.

With the data in the proper format it is now ready for pre-processing. My original approach as I was thinking about how to best implement this project was to simply make a term-frequency matrix, plot the points, cluster the points, predict new points. However, after reviewing the size of the data it became clear that a smarter way would be needed if the processing were to take place in any reasonable amount of time.

To get my implementation of the smarter way started I studied an example in the sklearn documentation (the first entry in the COLLABORATORS file) which went through document clustering using latent semantic analysis and kmeans. First the data is vectorized using TfidfVectorizer from sklearn. The TfidfVectorizer determines the importance of a given word to a text by measuring how frequently the word appears in a single document, compared to how frequently it appears in all the documents being used for training. With our recipes example each ingredient only appears in any given recipe one time, so the term frequency in a given recipe will be the same for all the ingredients in that recipe, however, the document frequency can be very important because if, say, garlic is in 99% of the recipes then garlic is not going to be very helpful in determining the cuisine of that recipe.

The Tfidf Vectorization process will produce an enormous matrix and trying to get kmeans to act on it in a consistent and accurate way will take a very long time. However, means exist by which the dimensionality of matrices can be reduced. Sklearn documentation recommends using TruncatedSVD to reduce the dimensionality of sparse matrices By using sklearn's truncatedSVD method which, according to the method's documentation in the sklearn website "In particular, truncatedSVD works on tf-ift matrices as returned by the vectorizers in sklearn.feature\_extraction.text. In that context it is known as latent semantic analysis (LSA)."

Finally, since the results of teh TruncatedSVD process are not normalized, the matrix is normalized so that kmeans can work most effectively.

With the pre-processing complete, building a kmeans model using the sklearn kmeans method is a simple and quick matter. With the kmeans model fully in place, new data is taken from the user, subjected to the same SLA treatment as the original data and then a prediction is made on the cluster that the new data should be assigned to.

The output indicates the cuisine of the supplied ingredients and a number of recommended similar dishes, where the number matches the N tag supplied by the user at the time the program was ran.

### 4. TESTS

Tests to verify operability are amply supplied.

There are only two methods in this implemenation, and thus only two tests.

The first test demonstrates that a list of ingredients is properly turned into a string of ingredients.

The second test runs the whole program with a sample input that is clearly a chinese dish (soy sauce, rice, eggs, shrimp) and asks for 5 recommended dishes in return. Then we test to ensure that the returned cuisine is indeed chinese and that the number of recommended dishes provided was 5.

### 5. OTHER NOTES

Various websites were consulted including sklearn docs and othe rsites used in order to enhance my understanding of the sklearn package and the theory that it is based on. With this knowledge I designed my own implementation of this project. All consulted sites are listed in the COLLABORATORS file.
