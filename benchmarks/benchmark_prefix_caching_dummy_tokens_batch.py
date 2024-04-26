import argparse
import time

from vllm import LLM, SamplingParams
import numpy as np

PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"  # noqa: E501


def test_prefix(llm=None, sampling_params=None, prompts=None):
    start_time = time.time()

    llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)

    end_time = time.time()
    print(f"cost time {(end_time - start_time)* 1000} ")


def main(args):
    llm = LLM(model="/home/jovyan/models/Llama-2-13b-hf/",
              tokenizer_mode='auto',
              trust_remote_code=True,
              enforce_eager=True,
              enable_prefix_caching=args.enable_prefix_caching,
              enable_radix_caching=args.enable_radix_caching,
              block_size=args.block_size)

    # num_prompts = 100
    # prompts = [PROMPT] * num_prompts
    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)

    
    cache_len = int(args.input_len * args.cache_ratio)
    cached_len =  int(cache_len * args.cached_pre_ratio)
    dummy_prompt_cache_token_ids = np.random.randint(0, 1,
                                               size=(args.batch_size,
                                                     cache_len))
    dummy_prompt_cache_token_ids = dummy_prompt_cache_token_ids.tolist()
    
    input_len = args.input_len - cache_len
    dummy_prompt_token_ids = np.random.randint(0, 1,
                                               size=(args.batch_size,
                                                     cache_len))
    dummy_prompt_token_ids = dummy_prompt_token_ids.tolist()
    
    dummy_prompt_no_cache_token_ids1 = np.random.randint(1, 2,
                                               size=(args.batch_size,
                                                     cache_len))
    dummy_prompt_no_cache_token_ids1 = dummy_prompt_no_cache_token_ids1.tolist()
    
    dummy_prompt_no_cache_token_ids2 = np.random.randint(2, 3,
                                               size=(args.batch_size,
                                                     cache_len))
    dummy_prompt_no_cache_token_ids2 = dummy_prompt_no_cache_token_ids2.tolist()
    
    dummy_prompt_no_cache_token_ids3 = np.random.randint(3, 4,
                                               size=(args.batch_size,
                                                     cache_len))
    dummy_prompt_no_cache_token_ids3 = dummy_prompt_no_cache_token_ids3.tolist()
    
    print("------warm up------")
    test_prefix(
        llm=llm,
        prompts=[dummy_prompt_cache_token_ids[0][:cached_len]],
        sampling_params=sampling_params,
    )

    print("------start generating------")
    test_prefix(
        llm=llm,
        prompts=[dummy_prompt_token_ids[0][:cached_len]],
        sampling_params=sampling_params,
    )
    
    test_prefix(
        llm=llm,
        prompts=[dummy_prompt_no_cache_token_ids1[0][:cached_len]],
        sampling_params=sampling_params,
    )
    
    # test_prefix(
    #     llm=llm,
    #     prompts=[dummy_prompt_no_cache_token_ids2[0][:cached_len]],
    #     sampling_params=sampling_params,
    # )
    # test_prefix(
    #     llm=llm,
    #     prompts=[dummy_prompt_no_cache_token_ids3[0][:cached_len]],
    #     sampling_params=sampling_params,
    # )
    
    # print("------start generating------")
    test_prefix(
        llm=llm,
        # prompts=[dummy_prompt_token_ids[0], dummy_prompt_no_cache_token_ids1[0], dummy_prompt_no_cache_token_ids2[0], dummy_prompt_no_cache_token_ids3[0]],
        prompts=[dummy_prompt_token_ids[0], dummy_prompt_no_cache_token_ids1[0]],
        sampling_params=sampling_params,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark the performance with or without automatic '
        'prefix caching.')
    parser.add_argument('--enable-prefix-caching',
                        action='store_true',
                        help='enable prefix caching')
    parser.add_argument('--enable-radix-caching',
                        action='store_true',
                        help='enable prefix caching')
    parser.add_argument('--cache-ratio', type=float, default=1)
    parser.add_argument('--input-len', type=int, default=1024)
    parser.add_argument('--output-len', type=int, default=1)
    parser.add_argument('--block-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)

    parser.add_argument('--cached-pre-ratio', type=float, default=1)



    args = parser.parse_args()
    main(args)
