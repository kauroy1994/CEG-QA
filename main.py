#import statements
import argparse
import logging
from definitions import FileIO, Trainer

def run():
    """
    main method for program
    """
    parser = argparse.ArgumentParser() #create argument parser
    parser.add_argument("--datapath") #create argument for datapath
    parser.add_argument("--datafile") #create argument for datafile
    parser.add_argument("--baseline") #create argument for baseline embedding method
    args = parser.parse_args() #load arguments
    try:
        assert not args.datapath is None #make sure datapath has been specified
        assert not args.datafile is None #make sure datafile is specified
        assert args.baseline in [None,"trans_e","dm_e","comp_e","hyp_e"]
    except AssertionError as error:
        logging.error("Either Datapath or datafile incorrectly specified, (use --datapath <datapath> --datafile <datafile>)\n"
                      "or baseline specification error, (use --baseline with trans_e, dm_e, comp_e, or hyp_e)\n"
                      "valid run e.g.: python main.py --datapath <datapath> --datafile <datafile> --baseline trans_e\n"
                      "exiting program with error %s ... \n", error) #log error
        exit() #exit program
    data = FileIO.read_pickle_file(args.datapath,args.datafile) #read the datafile from the datapath
    trainer_config = FileIO.load_from_json() #load trainer config from train_config.json
    trainer = Trainer(trainer_config) #initialize trainer object with the trainer config
    trainer.load_training_data(data, baseline = (args.baseline if not args.baseline is None else None)) #load the training data
    model = trainer.train(epochs=50) #train the model with set number of epochs
if __name__ == "__main__":

    run() #main method
