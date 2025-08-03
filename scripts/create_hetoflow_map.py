import random as rand

RANK_CPU = "C0"


def create_default_expert_map(num_experts, tp_size, num_gpu_experts, random):
    ep_size = tp_size
    expert_map_plan = [0] * num_experts
    gpu_in_high_part = False
    if num_gpu_experts < 0:
        gpu_in_high_part = True
        gpu_expert_total = -num_gpu_experts
    else:
        gpu_expert_total = num_gpu_experts

    if not random:
        print(f"create_default_expert_map, gpu_expert_total:{gpu_expert_total}")
        for e_id in range(num_experts):
            if (
                num_experts - 1 - e_id if gpu_in_high_part else e_id
            ) < gpu_expert_total:
                expert_map_plan[e_id] = e_id % ep_size
            else:
                expert_map_plan[e_id] = RANK_CPU
    else:
        print(f"Creating random expert map, gpu_expert_total:{gpu_expert_total}")
        gpu_experts = rand.sample(
            range(num_experts), min(gpu_expert_total, num_experts)
        )
        gpu_expert_set = set(gpu_experts)

        # Distribute GPU experts evenly across ranks
        gpu_expert_list = list(gpu_experts)
        for i, e_id in enumerate(gpu_expert_list):
            expert_map_plan[e_id] = i % ep_size

        # Assign remaining experts to CPU
        for e_id in range(num_experts):
            if e_id not in gpu_expert_set:
                expert_map_plan[e_id] = RANK_CPU

    return expert_map_plan


def create_model_expert_map(
    num_experts, tp_size, num_gpu_experts, layer_start, layer_end, random
):
    model_expert_plan = dict()
    for l in range(layer_start, layer_end):
        layer_map_plan = create_default_expert_map(
            num_experts, tp_size, num_gpu_experts, random
        )
        model_expert_plan[l] = layer_map_plan
    return model_expert_plan


def dump_expert_map_json(expert_map, file_path):
    import json

    with open(file_path, "w") as f:
        json.dump(expert_map, f, indent=4)
    print(f"Expert map dumped to {file_path}")


def create_plan(
    model_name, layer_start, layer_end, num_experts, tp_size, num_gpu_experts, random
):
    expert_map = create_model_expert_map(
        num_experts, tp_size, num_gpu_experts, layer_start, layer_end, random
    )
    file_path = f"{model_name}_expert_map_gpu_{num_gpu_experts}_tp_{tp_size}{'_random' if random else ''}.json"
    dump_expert_map_json(expert_map, file_path)
    return expert_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create default expert map")
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the model"
    )
    parser.add_argument(
        "--layer-start", type=int, default=0, help="Starting layer index"
    )
    parser.add_argument("--layer-end", type=int, default=0, help="Ending layer index")
    parser.add_argument(
        "--num-experts", type=int, required=True, help="Total number of experts"
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--num-gpu-experts",
        type=int,
        default=-1,
        help="Number of GPU experts (negative for high part)",
    )
    parser.add_argument("--random", action="store_true", help="Enable random mode")

    args = parser.parse_args()

    create_plan(
        args.model_name,
        args.layer_start,
        args.layer_end,
        args.num_experts,
        args.tp_size,
        args.num_gpu_experts,
        args.random,
    )
