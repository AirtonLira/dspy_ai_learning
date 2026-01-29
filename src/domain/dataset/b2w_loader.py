import os
import pandas as pd
from pathlib import Path
from utils.config import get_data_path


class B2WLoader:
    """Responsável por fazer download e carregar o dataset B2W Reviews"""
    
    def __init__(self, path: str = None):
        self.path = path if path else get_data_path()
        self.df = None
    
    def load(self) -> pd.DataFrame:
        """Carrega o CSV do dataset B2W"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Dataset não encontrado em: {self.path}\n"
                f"Verifique se o arquivo CSV existe ou faça o download manualmente."
            )
        
        print(f"Carregando dataset de: {self.path}")
        self.df = pd.read_csv(self.path)
        print(f"Dataset carregado: {len(self.df)} registros")
        
        return self.df
    
    def get_dataframe(self) -> pd.DataFrame:
        """Retorna o dataframe carregado"""
        if self.df is None:
            self.load()
        return self.df
    
    @staticmethod
    def download_dataset(destination: str = None) -> str:
        """
        Faz download do dataset B2W Reviews (se necessário)
        
        Retorna o caminho do arquivo baixado
        """
        dest_path = destination if destination else get_data_path()
        
        print(f"Para usar o dataset B2W Reviews, faça download em:")
        print(f"https://www.kaggle.com/datasets/b2w-reviews")
        print(f"\nSalve o arquivo CSV em: {dest_path}")
        
        return dest_path