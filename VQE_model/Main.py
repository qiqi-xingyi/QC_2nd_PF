# --*-- conding:utf-8 --*--
# @Time : 1/17/25 11:14 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : Main.py

import os
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem
from qiskit_ibm_runtime import QiskitRuntimeService
from Qiskit_VQE import VQE5
from Qiskit_VQE import StateCalculator
import time


def predict_protein_structure(
    main_chain_sequence: str,
    protein_id: str,
    service: QiskitRuntimeService,
    max_iter: int = 150
):
    """
    Use the given quantum VQE workflow to predict a protein structure based on the
    specified amino acid sequence, and store the results in corresponding directories.

    :param main_chain_sequence: The main chain amino acid sequence (single-letter representation).
    :param protein_id: The identifier/name for the protein (used to name output files).
    :param service: An instance of QiskitRuntimeService for submitting quantum jobs.
    :param max_iter: Maximum iteration count for VQE, default=150.
    """

    print(f"Starting prediction for protein: {protein_id}, sequence: {main_chain_sequence}")

    # Create side chain sequences (empty in this demo)
    side_chain_sequences = ['' for _ in range(len(main_chain_sequence))]

    # Print basic information
    chain_length = len(main_chain_sequence)
    print(f"Number of amino acids: {chain_length}")

    side_chain_count = len(side_chain_sequences)
    print(f"Number of side chain sites: {side_chain_count}")

    peptide = Peptide(main_chain_sequence, side_chain_sequences)

    mj_interaction = MiyazawaJerniganInteraction()

    penalty_terms = PenaltyParameters(10, 10, 10)

    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)

    hamiltonian = protein_folding_problem.qubit_op()

    qubits_num = hamiltonian.num_qubits + 5
    print(f"Number of qubits: {qubits_num}")


    vqe_instance = VQE5(
        service=service,
        hamiltonian=hamiltonian,
        min_qubit_num=qubits_num,
        maxiter=max_iter
    )

    energy_list, res, ansatz, top_results = vqe_instance.run_vqe()

    output_energy_path = f"Result/process_data/best_group/{protein_id}/System_Enegry"
    os.makedirs(output_energy_path, exist_ok=True)
    with open(f"{output_energy_path}/energy_list_{protein_id}.txt", 'w') as file:
        for item in energy_list:
            file.write(str(item) + '\n')

    state_calculator = StateCalculator(service, qubits_num, ansatz)
    probability_distribution = state_calculator.get_probability_distribution(res)

    protein_result = protein_folding_problem.interpret(probability_distribution)

    output_prob_path = f"Result/process_data/best_group/{protein_id}/Prob_distribution"
    os.makedirs(output_prob_path, exist_ok=True)
    with open(f"{output_prob_path}/prob_distribution.txt", 'w') as file:
        for key, value in probability_distribution.items():
            file.write(f'{key}: {value}\n')

    output_dir = f"Result/process_data/best_group/{protein_id}"
    os.makedirs(output_dir, exist_ok=True)
    protein_result.save_xyz_file(name=protein_id, path=output_dir)
    print("Protein structure saved as .xyz file")

    for rank, (energy_val, best_params) in enumerate(top_results, start=1):
        print(f"Top {rank} best energy = {energy_val}")

        prob_dist_best = state_calculator.get_probability_distribution(best_params)
        protein_result_best = protein_folding_problem.interpret(prob_dist_best)
        protein_result_best.save_xyz_file(
            name=f"{protein_id}_top_{rank}",
            path=output_dir
        )
        print(f"Protein structure for top {rank} best result has been saved.")

    print(f"Finished processing: {protein_id} \n")


if __name__ == '__main__':

    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance=' ',  # Replace with your real instance
        token=' '      # Replace with your real token
    )

    protein_list = [
        ("DGKMKGLAF", "1qin"),
        ("IHGIGGFI",  "1a9m"),
        ("KSIVDSGTTNLR" , "1fkn"),
        ("NNLGTIAKSGT", "3b26"),
        ("GAVEDGATMTFF", "2xxx"),
        ("DWGGM", "3ans"),
        ("YAGYS", "6mu3")
    ]


    log_file_path = "execution_time_log.txt"
    with open(log_file_path, 'w') as log_file:
        log_file.write("Protein_ID\tExecution_Time(s)\n")

        for sequence, protein_name in protein_list:

            start_time = time.time()

            predict_protein_structure(
                main_chain_sequence=sequence,
                protein_id=protein_name,
                service=service,
                max_iter=150
            )

            end_time = time.time()
            execution_time = end_time - start_time
            log_file.write(f"{protein_name}\t{execution_time:.2f}\n")