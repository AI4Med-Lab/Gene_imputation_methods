import argparse
import os


class Options():

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # =====================================================
        # DATA
        # =====================================================

        self.parser.add_argument(
            '--root',
            type=str,
            default='./dataset/'
        )

        self.parser.add_argument(
            '--dataset_name',
            type=str,
            default='Dataset1'
        )

        # -----------------------------------------------------
        # ADD THESE TWO
        # -----------------------------------------------------

        self.parser.add_argument(
            '--seq_data',
            type=str,
            default=None,
            help='Path to scRNA-seq h5ad file'
        )

        self.parser.add_argument(
            '--spa_data',
            type=str,
            default=None,
            help='Path to spatial transcriptomics h5ad file'
        )

        # =====================================================
        # TRAINING
        # =====================================================

        self.parser.add_argument(
            '--kfold',
            type=int,
            default=0
        )

        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=512
        )

        self.parser.add_argument(
            '--epochs',
            type=int,
            default=200
        )

        self.parser.add_argument(
            '--val_only',
            type=int,
            default=0
        )

        self.parser.add_argument(
            '--style_dim',
            type=int,
            default=1
        )

        self.parser.add_argument(
            '--lr',
            type=float,
            default=1e-3
        )

        self.parser.add_argument(
            '--beta1',
            type=float,
            default=0.9
        )

        self.parser.add_argument(
            '--beta2',
            type=float,
            default=0.999
        )

        self.parser.add_argument(
            '--save_path',
            type=str,
            default='./SpaIM_results/'
        )

        self.parser.add_argument(
            '--load_path',
            type=str,
            default='best_pcc.pth'
        )

        self.parser.add_argument(
            '--seed',
            type=int,
            default=1234
        )

        self.parser.add_argument(
            '--parallel',
            type=int,
            default=0
        )

        self.parser.add_argument(
            '--cluster',
            type=str,
            default='annotate'
        )

        self.parser.add_argument(
            '--gpu',
            type=int,
            default=0
        )

        self.parser.add_argument(
            '--model_layers',
            type=str,
            default='256,512'
        )
        
        self.parser.add_argument(
        '--result_dir',
        type=str,
        default='results/SpaIM'
        )


    def parse(self):

        self.opt = self.parser.parse_args()

        # =====================================================
        # DATA PATHS
        # =====================================================
        #
        # If explicit paths are NOT supplied,
        # fall back to original SpaIM folder structure.
        #

        if self.opt.seq_data is None:

            self.opt.seq_data = os.path.join(
                self.opt.root,
                self.opt.dataset_name,
                'scRNA_count_cluster.h5ad'
            )

        if self.opt.spa_data is None:

            self.opt.spa_data = os.path.join(
                self.opt.root,
                self.opt.dataset_name,
                'Insitu_count.h5ad'
            )

        # =====================================================
        # CHECK FILES
        # =====================================================

        if not os.path.exists(
            self.opt.seq_data
        ):

            raise FileNotFoundError(
                f"scRNA data not found:\n"
                f"{self.opt.seq_data}"
            )

        if not os.path.exists(
            self.opt.spa_data
        ):

            raise FileNotFoundError(
                f"Spatial data not found:\n"
                f"{self.opt.spa_data}"
            )

        print("=" * 80)

        print(
            "Dataset       :",
            self.opt.dataset_name
        )

        print(
            "scRNA data    :",
            self.opt.seq_data
        )

        print(
            "Spatial data  :",
            self.opt.spa_data
        )

        print(
            "Fold          :",
            self.opt.kfold + 1
        )

        print("=" * 80)

        # =====================================================
        # SAVE PATH
        # =====================================================

        # User gives:
        #
        # --save_path checkpoints/SpaIM
        #
        # Result:
        #
        # checkpoints/SpaIM/seqFISH/

        self.opt.save_path = os.path.join(
            self.opt.save_path,
            self.opt.dataset_name
        )

        os.makedirs(
            self.opt.save_path,
            exist_ok=True
        )

        # =====================================================
        # MODEL LAYERS
        # =====================================================

        model_layers = (
            self.opt.model_layers
            .split(',')
        )

        self.opt.model_layers = [
            int(x)
            for x in model_layers
        ]

        return self.opt