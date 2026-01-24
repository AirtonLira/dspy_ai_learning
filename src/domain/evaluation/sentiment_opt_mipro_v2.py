"""
Sentiment Analysis Optimization using MIPROv2

Este módulo otimiza prompts para classificação de sentimento
usando a estratégia MIPROv2 do DSPy.

Autor: Airton Lira
Data: 2026-01-24
"""

import dspy
from dspy.teleprompt import MIPROv2
from domain.module.sentiment import SentimentClassifier
from domain.evaluation.sentiment_eval import sentiment_dataset
from pathlib import Path
import json
from datetime import datetime


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class SentimentMiproManager:
    """
    Gerenciador de otimização MIPROv2 para classificação de sentimento.
    
    Atributos:
        trainset: Dataset de treinamento
        base_program: Programa base SentimentClassifier
        compiled_program: Programa otimizado após compilação
    """
    
    def __init__(self):
        """Inicializa o gerenciador com dataset e programa base."""
        self.trainset = sentiment_dataset()
        self.base_program = None
        self.compiled_program = None
        
        if not self.trainset:
            print("❌ Erro: Dataset vazio!")
            return
        
        self.base_program = SentimentClassifier()
        print(f"✅ Gerenciador inicializado com {len(self.trainset)} exemplos")


    def _metric(self, example, pred, trace=None):
        """
        Métrica de avaliação: acurácia simples.
        
        Compara o sentimento predito com o esperado (case-insensitive).
        
        Args:
            example: Exemplo do dataset com campo 'sentiment'
            pred: Predição do modelo com campo 'sentiment'
            trace: Trace do DSPy (não usado aqui)
            
        Returns:
            int: 1 se correto, 0 se incorreto
        """
        try:
            expected = example.sentiment.lower().strip()
            predicted = pred.sentiment.lower().strip()
            
            is_correct = int(expected == predicted)
            return is_correct
            
        except AttributeError as e:
            print(f"⚠️  Erro ao acessar campos: {e}")
            return 0


    def run_mipro_optimization(self):
        """
        Executa otimização MIPROv2 com modo automático.
        
        Usa auto="medium" para deixar o DSPy decidir automaticamente
        os parâmetros de otimização (num_candidates, num_trials).
        
        Returns:
            dspy.ChainOfThought: Programa compilado otimizado ou None se falhar
        """
        
        if self.base_program is None:
            print("❌ Erro: Programa base não inicializado")
            return None
        
        if self.trainset is None or len(self.trainset) == 0:
            print("❌ Erro: Dataset vazio ou não carregado")
            return None
        
        print("\n" + "="*60)
        print("🚀 INICIANDO OTIMIZAÇÃO MIPROV2")
        print("="*60)
        print(f"   📊 Dataset: {len(self.trainset)} exemplos")
        print(f"   ⚙️  Modo: auto='medium'")
        print(f"   📈 Métrica: Acurácia (sentiment match)")
        print("="*60 + "\n")
        
        try:
            # ✅ CORRIGIDO: usar auto="medium" SEM num_candidates
            # Quando auto é fornecido, DSPy controla tudo automaticamente
            teleprompter = MIPROv2(
                prompt_model=dspy.settings.lm,
                task_model=dspy.settings.lm,
                metric=self._metric,
                auto="medium"  # ✅ Controla num_candidates e num_trials automaticamente
            )
            
            print("⏳ Compilando e otimizando programa...")
            print("   (Este processo pode levar alguns minutos...)\n")
            
            compiled_program = teleprompter.compile(
                student=self.base_program,
                trainset=self.trainset,
                max_bootstrapped_demos=2,
                max_labeled_demos=2
            )
            
            self.compiled_program = compiled_program
            
            print("\n" + "="*60)
            print("✅ OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
            print("="*60 + "\n")
            
            return compiled_program
            
        except ValueError as e:
            print(f"\n❌ ValueError durante otimização:")
            print(f"   {str(e)}\n")
            return None
            
        except Exception as e:
            print(f"\n❌ Erro inesperado: {type(e).__name__}")
            print(f"   {str(e)}\n")
            return None


    def evaluate_compiled_program(self):
        """
        Avalia o programa compilado no dataset de treinamento.
        
        Processa cada exemplo do dataset e calcula acurácia geral.
        
        Returns:
            dict: Dicionário com estatísticas de avaliação ou None se falhar
        """
        
        if self.compiled_program is None:
            print("❌ Erro: Programa compilado não existe")
            return None
        
        if self.trainset is None or len(self.trainset) == 0:
            print("❌ Erro: Dataset vazio")
            return None
        
        scores = []
        
        print("\n" + "="*60)
        print("📊 AVALIANDO PROGRAMA COMPILADO")
        print("="*60 + "\n")
        
        for i, example in enumerate(self.trainset):
            try:
                # Fazer predição
                prediction = self.compiled_program(text=example.text)
                
                # Calcular score
                score = self._metric(example, prediction)
                scores.append(score)
                
                # Progresso a cada 10 exemplos
                if (i + 1) % 10 == 0:
                    current_acc = sum(scores[:i+1]) / (i + 1)
                    print(f"   ✓ {i + 1:3d}/{len(self.trainset)} exemplos | "
                          f"Acurácia parcial: {current_acc:.2%}")
                    
            except Exception as e:
                print(f"   ⚠️  Erro ao processar exemplo {i}: {type(e).__name__}")
                scores.append(0)
        
        # Calcular acurácia final
        accuracy = sum(scores) / len(scores) if scores else 0
        correct = sum(scores)
        incorrect = len(scores) - correct
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "total_examples": len(scores),
            "correct": correct,
            "incorrect": incorrect,
            "accuracy_percentage": f"{accuracy:.2%}"
        }
        
        print("\n" + "="*60)
        print("📈 RESULTADOS DA AVALIAÇÃO")
        print("="*60)
        print(f"   Acurácia:      {accuracy:.2%}")
        print(f"   Corretos:      {correct}/{len(scores)}")
        print(f"   Incorretos:    {incorrect}/{len(scores)}")
        print("="*60 + "\n")
        
        return results


    def save_checkpoint(self, filename="sentiment_mipro_optimized.json"):
        """
        Salva o programa compilado em arquivo.
        
        Args:
            filename (str): Nome do arquivo de saída
            
        Returns:
            bool: True se sucesso, False se falhar
        """
        
        if self.compiled_program is None:
            print("❌ Erro: Nenhum programa compilado para salvar")
            return False
        
        try:
            filepath = RESULTS_DIR / filename
            
            print(f"💾 Salvando programa compilado...")
            self.compiled_program.save(str(filepath))
            print(f"✅ Programa salvo em: {filepath}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar programa: {type(e).__name__}: {e}\n")
            return False


    def save_results(self, results, filename="mipro_results.json"):
        """
        Salva resultados de avaliação em formato JSON.
        
        Args:
            results (dict): Dicionário com resultados
            filename (str): Nome do arquivo de saída
            
        Returns:
            bool: True se sucesso, False se falhar
        """
        
        if results is None:
            print("⚠️  Aviso: Resultados vazios, pulando salvamento")
            return False
        
        try:
            filepath = RESULTS_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Resultados salvos em: {filepath}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {type(e).__name__}: {e}")
            return False


def run_optimization():
    """
    Função principal para executar otimização MIPROv2.
    
    Orquestração completa:
    1. Inicializa gerenciador com dataset
    2. Executa otimização MIPROv2
    3. Avalia programa otimizado
    4. Salva resultados e checkpoint
    """
    
    print("\n" + "#"*60)
    print("# OTIMIZAÇÃO DE SENTIMENTO COM MIPROV2")
    print("#"*60 + "\n")
    
    # Inicializar gerenciador
    manager = SentimentMiproManager()
    
    if manager.base_program is None:
        print("❌ Falha na inicialização. Encerrando.")
        return
    
    # Executar otimização
    optimized_program = manager.run_mipro_optimization()
    
    if optimized_program is None:
        print("❌ Falha na otimização. Encerrando.")
        return
    
    # Avaliar programa compilado
    results = manager.evaluate_compiled_program()
    
    if results:
        # Salvar resultados
        manager.save_results(results)
        manager.save_checkpoint()
        
        print("\n" + "#"*60)
        print("# ✅ OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
        print("#"*60)
        print(f"   Acurácia Final: {results['accuracy_percentage']}")
        print(f"   Corretos: {results['correct']}/{results['total_examples']}")
        print("#"*60 + "\n")
    else:
        print("\n⚠️  Avaliação não retornou resultados válidos")


if __name__ == "__main__":
    # Para testes diretos
    import os
    from dotenv import load_dotenv
    from utils.config import setup_llm
    
    load_dotenv()
    setup_llm()
    
    run_optimization()
