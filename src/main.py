from models.particle_transformer import  *
from trainers.particle_classifcation_trainer import *
from data.top_classification_data_modules import *
from data.top_quark_reconstruction import *
from trainers.top_reconstruction_trainers import *
from utils.utils import load_and_split_config, load_any_config







if __name__ == "__main__":
    config = load_any_config("config/top_reconstruction_config.yaml")



    particle_embedder = ParticleEmbedder(**config["model_parameters"]["particle_embedder"])
    masked_prediction_head = ReverseEmbedder(**config["model_parameters"]["masked_prediction_head"])
    kinematic_regression_head = ReverseEmbedder(**config["model_parameters"]["kinematic_regression_head"])
    
    interactions_embedder = InteractionEmbedder(**config["model_parameters"]["interaction_embedder"])
   
    transformer_model = MaskedReconstructionPart(particle_embedder,interactions_embedder
                                                            , masked_prediction_head,  kinematic_regression_head , 
                                                            **config["model_parameters"]["transformer"], 
                                                            reconstruct_Ws = True,
                                                            use_hungarian_matching= config["use_hungarian_matching"])
    topantitopquark = TopandWReconstuctionDataModule(config)        
        

  
    
    print(config["reconstruct_W"])
    trainer, model = train_reconstruction_model(transformer_model,
                                                topantitopquark,
                                                config,
                                                5, 20,
                                                reconstruct_Ws = config["reconstruct_W"])
    trainer.test(model, datamodule = topantitopquark)

