# ResearchProjectRelationExtraction

The seed dataset in contained [here](https://github.com/romainremyb/ResearchProjectRelationExtraction/tree/main/rawdata). Labelled datasets for both NER and relation extraction tasks are displayed [there](https://github.com/romainremyb/ResearchProjectRelationExtraction/tree/main/labelledData).

This [file](https://github.com/romainremyb/ResearchProjectRelationExtraction/blob/main/unsupervised_methods.py) contains the unsupervised methods developed at the beginning of the project to extract labelled data from raw texts. The [notebook](https://github.com/romainremyb/ResearchProjectRelationExtraction/blob/main/unsupervised_relationExploration.ipynb) make use of these methods and also displays the number of manually labeled relations.

The NER [folder](https://github.com/romainremyb/ResearchProjectRelationExtraction/tree/main/ner) contains all files related to Named-entity-recognition tasks. Two pre-processing notebooks were implemented. The one called "merged" differs from the other in that it combined product and market entities (often embiguous). Fine-tuning of Bert for token classification can be found [here](https://github.com/romainremyb/ResearchProjectRelationExtraction/blob/main/ner/NERmodels.ipynb). It also contains hyperparameter fine-tuning but unfortunately does not display optimum parameters.

This [file](https://github.com/romainremyb/ResearchProjectRelationExtraction/blob/main/fine_tune_maskedModel.ipynb) contains the fine tuning of the bert model and this [notebook](https://github.com/romainremyb/ResearchProjectRelationExtraction/blob/main/rel_preprocessing.ipynb) describes the preprocessing procedure.

[Here](https://github.com/romainremyb/ResearchProjectRelationExtraction/blob/main/relation_classification.ipynb) is an implementation of a k-nearest-neighbor classifier and draws the baseline for new relation retrieval.

Finally, this [notebook](https://github.com/romainremyb/ResearchProjectRelationExtraction/blob/main/predict_relations_pipeline.ipynb) implements the whole pipeline (NER+classification) to predict new relations from text inputs.
