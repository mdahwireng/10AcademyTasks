class TweetDFModelEvaluate:
    """
    this function will evaluate and select the best performing model and optimize its performance
    
    Return
    ------
    dict
    """
    
    def __init__ (self, models):
        self.models = models
        
    def evaluate_model(self):
        """
        this function will evaluate the trained models

        Return
        ------
        None
        """
        models = self.models
        for key in models.keys():
            X_train = models[key]['X_train']
            X_valid = models[key]['X_valid']
            y_train = models[key]['y_train']
            y_valid = models[key]['y_valid']
            model = models[key]['model']
            train_score = model.score(X_train, y_train)
            valid_score = model.score(X_valid, y_valid)
            models[key]['train_score'] = train_score
            models[key]['valid_score'] = valid_score
            
        self.models =  models
        
    def select_model(self):
        """
        this function will select the best performing trained model

        Return
        ------
        dict
        """
        models = self.models
        max_valid_score = 0
        selected_model = ''
        num_of_models = len(models.keys())
        print('The number of trained models are {}'.format(num_of_models))
        for key in models.keys():
            train_score = models[key]['train_score']
            valid_score = models[key]['valid_score']
            print('\nmodel name : {}\nTraining Score : {}\nValidation Score : {}'.format(key, train_score, valid_score))

            if models[key]['valid_score'] > max_valid_score:
                max_valid_score = models[key]['valid_score']
                selected_model = {'name': key, 'train_score':train_score, 'valid_score':valid_score, 'model': models[key]['model']}

        print('\nThe best performing and selected model is {}'.format(selected_model['name']))
        self.models = selected_model
        return selected_model