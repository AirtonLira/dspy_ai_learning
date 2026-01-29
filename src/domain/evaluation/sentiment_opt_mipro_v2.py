# src/domain/evaluation/sentiment_opt_mipro_v2.py
import sys
import json
import random
import dspy
import dspy
import os
from datetime import datetime
from pathlib import Path
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
from domain.module.sentiment import SentimentClassifier
from domain.dataset.b2w_review import B2WReviews
from domain.evaluation.sentiment_eval import sentiment_accuracy
from utils.rate_limiter import gemini_rate_limiter

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class SentimentMiproManager:
    """Gerenciador de otimização MIPROv2 para análise de sentimento"""
    
    def __init__(self, sample: int = 0):
        print("Inicializando SentimentMiproManager...")
        
        try:
            # Carregar dataset
            b2w = B2WReviews(sample=sample)
            self.trainset, self.testset = b2w.get_train_test_split()
            
            if not self.trainset or not self.testset:
                raise ValueError("Erro: Dataset vazio após processamento!")
            
            # Programa base
            self.base_program = SentimentClassifier()
            self.compiled_program = None
            
            # Metadata
            total_size = len(self.trainset) + len(self.testset)
            self.optimization_results = {
                "timestamp": datetime.now().isoformat(),
                "model": "gemini/gemini-3-flash-preview",
                "dataset_size": total_size,
                "train_size": len(self.trainset),
                "test_size": len(self.testset),
            }
            
            print(f"✓ SentimentMiproManager inicializado com sucesso!")
            print(f"  Total: {total_size} exemplos")
            
        except Exception as e:
            print(f"✗ Erro ao inicializar: {str(e)}")
            raise

    def _metric(self, example, pred, trace=None):
        """Métrica de avaliação: acurácia exata"""
        return sentiment_accuracy(example, pred)

    @gemini_rate_limiter
    def run_mipro_optimization(self, num_candidates: int = 3):
        """Executa otimização MIPROv2 completa"""
        print("\n" + "="*70)
        print("INICIANDO OTIMIZACAO MIPROV2")
        print("="*70)
        
        # Avaliar baseline
        baseline_score = self._evaluate_baseline()
        
        # Otimizar com MIPROv2
        print("\nRodando MIPROv2...")
        teleprompter = MIPROv2(
            prompt_model=dspy.settings.lm,
            task_model=dspy.settings.lm,
            metric=self._metric,
            auto="light",
        )
        
        # Reduzindo num_trials para economizar cota de API
        self.compiled_program = teleprompter.compile(
            self.base_program,
            trainset=self.trainset,
            num_trials=5,
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            requires_permission_to_run=False
        )
        
        # Avaliar programa otimizado
        optimized_score = self._evaluate_optimized()
        
        # Salvar resultados
        self._save_all_results(baseline_score, optimized_score)
        
        # Mostrar comparação
        self._show_comparison(baseline_score, optimized_score)
        
        # Mostrar exemplos
        self._show_predictions()
        
        return self.compiled_program

    def _evaluate_baseline(self):
        """Avalia programa base (sem otimização)"""
        print("\nAvaliando BASELINE (sem otimizacao)...")
        
        evaluator = Evaluate(
            devset=self.testset,
            metric=self._metric,
            display_progress=True,
            display_table=False,
            num_threads=1
        )
        
        result = evaluator(self.base_program)
        score = self._extract_score(result)
        
        self.optimization_results["baseline_score"] = score
        print(f"Baseline: {score:.2%}")
        
        return score

    def _evaluate_optimized(self):
        """Avalia programa otimizado"""
        print("\nAvaliando OTIMIZADO (apos MIPROv2)...")
        
        evaluator = Evaluate(
            devset=self.testset,
            metric=self._metric,
            display_progress=True,
            display_table=False,
            num_threads=1
        )
        
        result = evaluator(self.compiled_program)
        score = self._extract_score(result)
        
        class LLMConfig:
            _instance = None
        
            @classmethod
            def get_instance(cls, model="llama3", base_url="http://localhost:11434"):
                """Retorna a conexão com o LLM em memória (Singleton)."""
                if cls._instance is None:
                    
                    gemini_api_key = os.getenv("GOOGLE_API_KEY")
                    llm_local_mode = os.getenv("DSPY_AI_LOCAL_MODE", "false").lower()
                    if llm_local_mode == "true":
                        llm = dspy.LM(
                            model="ollama/glm4:9b-chat-q3_K_M",
                            chat=True,
                            max_tokens=256,
                            local_mode=True
                        )
                        print("Usando modelo local (Ollama GLM4).")
                    elif gemini_api_key:
                        llm = dspy.LM(
                            model="gemini/gemini-1.5-flash",
                            api_key=gemini_api_key,
                            chat=True,
                            max_tokens=2048,
                            api_version="v1"
                        )
                        print("Usando modelo remoto (Google Gemini 1.5 Flash - v1).")
                    else:
                        llm = dspy.LM(
                            model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
                            api_key=os.getenv("OPENROUTER_API_KEY"),
                            chat=True,
                            max_tokens=256
                        )
                        print("Usando modelo remoto (Liquid LFM 2.5).")
                    
                    print(f"--- Inicializando conexao com Ollama ({model}) ---")
                    cls._instance = llm
                    dspy.settings.configure(lm=cls._instance)
                return cls._instance
        
        def setup_llm():
            return LLMConfig.get_instance()
        
        def get_data_path():
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(src_dir, 'domain', 'dataset', 'data', 'b2w_reviews.csv')         
        i   import dspy
            import os
            
            class LLMConfig:
                _instance = None
            
                @classmethod
                def get_instance(cls, model="llama3", base_url="http://localhost:11434"):
                    """Retorna a conexão com o LLM em memória (Singleton)."""
                    if cls._instance is None:
                        
                        gemini_api_key = os.getenv("GOOGLE_API_KEY")
                        llm_local_mode = os.getenv("DSPY_AI_LOCAL_MODE", "false").lower()
                        if llm_local_mode == "true":
                            llm = dspy.LM(
                                model="ollama/glm4:9b-chat-q3_K_M",
                                chat=True,
                                max_tokens=256,
                                local_mode=True
                            )
                            print("Usando modelo local (Ollama GLM4).")
                        elif gemini_api_key:
                            llm = dspy.LM(
                                model="gemini/gemini-1.5-flash",
                                api_key=gemini_api_key,
                                chat=True,
                                max_tokens=2048,
                                api_version="v1"
                            )
                            print("Usando modelo remoto (Google Gemini 1.5 Flash - v1).")
                        else:
                            llm = dspy.LM(
                                model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
                                api_key=os.getenv("OPENROUTER_API_KEY"),
                                chat=True,
                                max_tokens=256
                            )
                            print("Usando modelo remoto (Liquid LFM 2.5).")
                        
                        print(f"--- Inicializando conexao com Ollama ({model}) ---")
                        cls._instance = llm
                        dspy.settings.configure(lm=cls._instance)
                    return cls._instance
            
            def setup_llm():
                return LLMConfig.get_instance()
            
            def get_data_path():
                src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                return os.path.join(src_dir, 'domain', 'dataset', 'data', 'b2w_reviews.csv')                import dspy
                import os
                
                class LLMConfig:
                    _instance = None
                
                    @classmethod
                    def get_instance(cls, model="llama3", base_url="http://localhost:11434"):
                        """Retorna a conexão com o LLM em memória (Singleton)."""
                        if cls._instance is None:
                            
                            gemini_api_key = os.getenv("GOOGLE_API_KEY")
                            llm_local_mode = os.getenv("DSPY_AI_LOCAL_MODE", "false").lower()
                            if llm_local_mode == "true":
                                llm = dspy.LM(
                                    model="ollama/glm4:9b-chat-q3_K_M",
                                    chat=True,
                                    max_tokens=256,
                                    local_mode=True
                                )
                                print("Usando modelo local (Ollama GLM4).")
                            elif gemini_api_key:
                                llm = dspy.LM(
                                    model="gemini/gemini-1.5-flash",
                                    api_key=gemini_api_key,
                                    chat=True,
                                    max_tokens=2048,
                                    api_version="v1"
                                )
                                print("Usando modelo remoto (Google Gemini 1.5 Flash - v1).")
                            else:
                                llm = dspy.LM(
                                    model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
                                    api_key=os.getenv("OPENROUTER_API_KEY"),
                                    chat=True,
                                    max_tokens=256
                                )
                                print("Usando modelo remoto (Liquid LFM 2.5).")
                            
                            print(f"--- Inicializando conexao com Ollama ({model}) ---")
                            cls._instance = llm
                            dspy.settings.configure(lm=cls._instance)
                        return cls._instance
                
                def setup_llm():
                    return LLMConfig.get_instance()
                
                def get_data_path():
                    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    return os.path.join(src_dir, 'domain', 'dataset', 'data', 'b2w_reviews.csv')                    import dspy
                    import os
                    
                    class LLMConfig:
                        _instance = None
                    
                        @classmethod
                        def get_instance(cls, model="llama3", base_url="http://localhost:11434"):
                            """Retorna a conexão com o LLM em memória (Singleton)."""
                            if cls._instance is None:
                                
                                gemini_api_key = os.getenv("GOOGLE_API_KEY")
                                llm_local_mode = os.getenv("DSPY_AI_LOCAL_MODE", "false").lower()
                                if llm_local_mode == "true":
                                    llm = dspy.LM(
                                        model="ollama/glm4:9b-chat-q3_K_M",
                                        chat=True,
                                        max_tokens=256,
                                        local_mode=True
                                    )
                                    print("Usando modelo local (Ollama GLM4).")
                                elif gemini_api_key:
                                    llm = dspy.LM(
                                        model="gemini/gemini-1.5-flash",
                                        api_key=gemini_api_key,
                                        chat=True,
                                        max_tokens=2048,
                                        api_version="v1"
                                    )
                                    print("Usando modelo remoto (Google Gemini 1.5 Flash - v1).")
                                else:
                                    llm = dspy.LM(
                                        model="openrouter/liquid/lfm-2.5-1.2b-instruct:free",
                                        api_key=os.getenv("OPENROUTER_API_KEY"),
                                        chat=True,
                                        max_tokens=256
                                    )
                                    print("Usando modelo remoto (Liquid LFM 2.5).")
                                
                                print(f"--- Inicializando conexao com Ollama ({model}) ---")
                                cls._instance = llm
                                dspy.settings.configure(lm=cls._instance)
                            return cls._instance
                    
                    def setup_llm():
                        return LLMConfig.get_instance()
                    
                    def get_data_path():
                        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        return os.path.join(src_dir, 'domain', 'dataset', 'data', 'b2w_reviews.csv')
        self.optimization_results["optimized_score"] = score
        print(f"Otimizado: {score:.2%}")
        
        return score

    def _extract_score(self, result):
        """Extrai score numérico do resultado"""
        score = 0.0
        if hasattr(result, 'score'):
            score = float(result.score)
        elif isinstance(result, (int, float)):
            score = float(result)
        
        # Se o score for > 1, assume que está em escala 0-100 e converte para 0-1
        if score > 1.0:
            score = score / 100.0
            
        return score

    def _show_comparison(self, baseline, optimized):
        """Mostra comparação entre baseline e otimizado"""
        improvement = optimized - baseline
        pct_improvement = (improvement / baseline * 100) if baseline > 0 else 0
        
        print("\n" + "="*70)
        print("COMPARACAO DE RESULTADOS")
        print("="*70)
        print(f"Baseline:     {baseline:.4f} ({baseline:.2%})")
        print(f"Otimizado:    {optimized:.4f} ({optimized:.2%})")
        print(f"\nMelhoria:     {improvement:+.4f} ({pct_improvement:+.1f}%)")
        print("="*70 + "\n")

    def _show_predictions(self):
        """Mostra exemplos de predições"""
        print("\n" + "="*70)
        print("EXEMPLOS DE PREDICOES")
        print("="*70)
        
        examples = random.sample(self.testset, min(3, len(self.testset)))
        
        for i, example in enumerate(examples, 1):
            print(f"\nExemplo {i}:")
            print(f"Texto: {example.review_text[:80]}...")
            
            pred_base = self.base_program(review_text=example.review_text)
            pred_opt = self.compiled_program(review_text=example.review_text)
            
            correct_base = example.sentiment.lower() == pred_base.sentiment.lower()
            correct_opt = example.sentiment.lower() == pred_opt.sentiment.lower()
            
            print(f"Real:      {example.sentiment}")
            print(f"Baseline:  {pred_base.sentiment} {'[✓]' if correct_base else '[✗]'}")
            print(f"Otimizado: {pred_opt.sentiment} {'[✓]' if correct_opt else '[✗]'}")
            print("-"*70)

    def _save_all_results(self, baseline_score, optimized_score):
        """Salva todos os resultados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        program_file = RESULTS_DIR / f"mipro_program_{timestamp}.json"
        self.compiled_program.save(program_file)
        print(f"\nPrograma salvo: {program_file}")
        
        results_file = RESULTS_DIR / "dspy_results.txt"
        self._save_results_txt(results_file, baseline_score, optimized_score)
        print(f"Resultados salvos: {results_file}")
        
        metadata_file = RESULTS_DIR / f"metadata_{timestamp}.json"
        self._save_metadata(metadata_file, baseline_score, optimized_score)
        print(f"Metadata salvo: {metadata_file}")

    def _save_results_txt(self, filepath, baseline_score, optimized_score):
        """Salva resultados em TXT"""
        improvement = optimized_score - baseline_score
        pct_improvement = (improvement / baseline_score * 100) if baseline_score > 0 else 0
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write("\n" + "="*70 + "\n")
            f.write("RESULTADOS DA OTIMIZACAO MIPROV2\n")
            f.write("="*70 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.optimization_results['dataset_size']} exemplos\n")
            f.write(f"  - Treino: {self.optimization_results['train_size']}\n")
            f.write(f"  - Teste:  {self.optimization_results['test_size']}\n\n")
            f.write(f"Baseline:   {baseline_score:.4f} ({baseline_score:.2%})\n")
            f.write(f"Otimizado:  {optimized_score:.4f} ({optimized_score:.2%})\n")
            f.write(f"Melhoria:   {improvement:+.4f} ({pct_improvement:+.1f}%)\n\n")

    def _save_metadata(self, filepath, baseline_score, optimized_score):
        """Salva metadata em JSON"""
        metadata = {
            **self.optimization_results,
            "baseline_score": baseline_score,
            "optimized_score": optimized_score,
            "improvement": optimized_score - baseline_score,
            "improvement_pct": ((optimized_score - baseline_score) / baseline_score * 100) 
                              if baseline_score > 0 else 0,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def save_checkpoint(self, filename: str = "sentiment_mipro_final.json"):
        """Salva checkpoint do programa otimizado"""
        if not self.compiled_program:
            print("✗ Erro: Nenhum programa compilado")
            return
        
        filepath = RESULTS_DIR / filename
        self.compiled_program.save(filepath)
        print(f"✓ Checkpoint salvo: {filepath}")