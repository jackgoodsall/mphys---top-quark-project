from models.particle_transformer import  *
from trainers.particle_classifcation_trainer import *
from data.top_classification_data_modules import *
from data.top_quark_reconstruction import *
from trainers.top_reconstruction_trainers import *
from utils.utils import load_and_split_config, load_any_config







if __name__ == "__main__":
    config = load_any_config("config/top_reconstruction_config.yaml")



    particle_embedder = ParticleEmbedder(**config["model_parameters"]["particle_embedder"])
    reverse_embedder = ReverseEmbedder(**config["model_parameters"]["reverse_embedder"])
    interactions_embedder = InteractionEmbedder(**config["model_parameters"]["interaction_embedder"])
   
    transformer_model = MaskedReconstructionPart(particle_embedder,interactions_embedder
                                                            , reverse_embedder, **config["model_parameters"]["transformer"], reconstruct_Ws = True,
                                                            use_hungarian_matching= config["use_hungarian_matching"])
    topantitopquark = TopandWReconstuctionDataModule(config)        
        

  
    
    print(config["reconstruct_W"])
    trainer, model = train_reconstruction_model(transformer_model,
                                                topantitopquark,
                                                config,
                                                5, 20,
                                                reconstruct_Ws = config["reconstruct_W"])
    trainer.test(model, datamodule = topantitopquark)


## Random place holder to remember the classication pipeline running 
if __name__ == "Fourtop_classifcation":
    config = load_and_split_config("config/transformer_classifier_config.yaml")
    classifer_head = ParticleBinaryClassificaitionHead(**config.model_parameters["classifer_head"])
    config.model_parameters["transformer"]["classifcation_head"] = classifer_head
    transformer_model = ParticleTransformer(**config.model_parameters["transformer"])
    fourtop_dataset = FourvsThreeTopDataModule(**config.data.train)
    trainer, model  = train_binary_classifier_model(transformer_model, 
                                   fourtop_dataset,
                                   config.train,
                                   config.train["min_epochs"],
                                   config.train["max_epochs"],
                                   )
    trainer.test(model, datamodule= fourtop_dataset)
