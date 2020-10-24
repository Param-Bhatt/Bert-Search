# Embeddings-Search
Implemented a searching algorithm to check a sentence through a preset database of emails and find out the most similar in meaning

### Guidelines for using :
* <i>dataset</i> is a global variable which contains all the data that our model will use. Hence, update the dataset variable.
<br>```dataset = set_dataset(<your_dataset>)```<br>
<i>Note that the dataset variable should be an array of strings.</i>
* Now , to run the program , run :<br>
                ```
                unprep, prep = load_dataset_and_preprocess(dataset)
                ```
     <br>
    * For BERT Queries , you will have to run :<br>
                ```
                unprep_index = build_bert_embeddings_index(unprep)
                ```
                <br>
                ```
                unprep_indx_to_email = build_reference(unprep)
                ```
    * For GLOVE Queries, you will have run : <br>
                ```
                embeddings_dict  = init_glove_embeddings()
                ```
                <br>
                ```
                prep_embeddings, prep_sentence = build_glove_index(prep)
                ```
* Once the dataset is fixed , you will have to rerun the entire code , since we are pre-processing and storing all the embedded data in our files.
* Now , to execute a query, call the <i>search_<i> function, depending on whether you want to execute search based on glove-embeddings or bert-embeddings. Since both will give accurate results , the choice is left on the user.
    * BERT Query (Normal data): <br>
      ```
      bert_search(query, unprep_index, unprep, unprep_indx_to_email, top_n=4, is_preprocess=False)
      ```
    * BERT Query (Pre-processed data) : <br>
      ```
      bert_search(query, prep_index, prep, prep_indx_to_email, top_n=4, is_preprocess=True)
      ```
    * GLOVE Query (Normal data): <br>
      ```
      glove_search(query, unprep_embeddings, unprep, unprep_indx_to_email, top_n=4, is_preprocess=False)
      ```
    * GLOVE Query (Pre-processed data) : <br>
      ```
      glove_search(query, prep_embeddings, prep, prep_indx_to_email, top_n=4, is_preprocess=True)
      ```
 ### Dependencies :
 We install these libraries each time when the program is run, and it is recommended to enable GPU on your google colab as well, to speed up the pre-processing.
 * datasets<br>
      * <i>We have used datasets for taking our dataset. If you have a custom dataset, you can omit this.</i>
 * nltk
 * numpy
 * pickle
 * mxnet *(Optional)* <br>
      * <i>It is recommended to use a GPU to speed up the pre-processing for your dataset. You can skip it, if you wish to.</i> <br> 
      * <i>Also, mxnet-cuda is specific to nvidia graphic cards. Please note this.</i> <br>
 * gluonnlp *(Optional, it is a dependency for mxnet)
 * bert-embedding
 * re(regex)
 * pdoc3 ( for documentation )
### Contribution Guidelines:

##### Creating a PR:
* Fork the repository to your own account
* Clone the repository to your local system , make any changes you wish to
```
git clone https://github.com/<your_username>/Search-Engine
```
* For each new feature, create a new branch of the name <i>feature_name<i>
  ```
  git checkout -b <feature_name>
  ```
* Comment the code properly with all necessary comments wherever needed
* While coding , follow the iflake coding guidelines for a cleaner and better code quality.
    * You can install iflake8 on your system as :
      ```
      pip install flake8
      ```
    * You can run it as :
      ```
      flake8 path/to/your/code
      ```
* Push the changes to your fork
* Create a pull request
* Resolve conflicts as required

##### To install dependencies
* datasets
```
pip install datasets
```
* nltk
```
pip install nltk
```
* numpy
```
pip install numpy
```
* pickle
```
pip install pickle5
```
* mxnet
```
pip install mxnet-cu101
```
* gluonnlp
```
pip install gluonnlp
```
* bert-embedding
```
pip install bert-embedding --no-deps
```
* re(regex)
```
pip install regex
```
