import os
import subprocess
import time
import json
import pandas as pd

from config import RESULTS_DIR, MODELS, AGGREGATION_STRATEGIES, PRIVACY_TECHNIQUES

class FederatedExperiment:

    def __init__(self):
        self.results = []

    def run_experiment(self, model_type, aggregation_strategy, privacy_technique, num_rounds=10):
        print(f"Ejecutando: {model_type} | {aggregation_strategy} | {privacy_technique}")
        os.environ["MODEL_TYPE"] = model_type
        os.environ["AGGREGATION_STRATEGY"] = aggregation_strategy
        os.environ["PRIVACY_TECHNIQUE"] = privacy_technique

        try:
            # ⏩ Ejecutar flwr run como subproceso
            subprocess.run([
                "flwr", "run", "federated.app:start"
            ], check=True)

            # ⏬ Leer métricas
            metrics_path = os.path.join(RESULTS_DIR, "last_metrics.json")
            if not os.path.exists(metrics_path):
                print("⚠️ No se generó last_metrics.json")
                return None

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            result = {
                'model_type': model_type,
                'aggregation_strategy': aggregation_strategy,
                'privacy_technique': privacy_technique,
                'num_rounds': num_rounds,
                **metrics
            }

            self.results.append(result)
            return result

        except Exception as e:
            print("❌ Error en experimento:", str(e))
            return None

    def run_all_experiments(self):
        total = len(MODELS) * len(AGGREGATION_STRATEGIES) * len(PRIVACY_TECHNIQUES)
        current = 1

        for model_type in MODELS:
            for strategy in AGGREGATION_STRATEGIES:
                for privacy in PRIVACY_TECHNIQUES:
                    print(f"\nProgreso: {current}/{total}")
                    self.run_experiment(model_type, strategy, privacy)
                    current += 1

        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(RESULTS_DIR, "resumen_resultados.csv"), index=False)
        print("\n✅ Benchmark generado correctamente.")


if __name__ == "__main__":
    experiment = FederatedExperiment()
    experiment.run_all_experiments()
