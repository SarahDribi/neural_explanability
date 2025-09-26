

"""
Some images have more reliable nap spec ,  they are more robust 

"""


"""
This function choses another epsilon within the epsilon range ,
and updates the value of current epsilon , the epsilon_range
"""
"""
The robustness of images depends on images and models 
the bigger the model the more robust it is
"""

def max_epsilon_robustness_binary_search(input,nap,low,high):
    return 1 # or less
    # 
def get_correctly_classified_inputs(label):

    # returns a list of tensors




def adjust_epsilon_according_to_passing_rates(passed_inputs,failed_inputs,epsilon_range,current_epsilon):
    num_passed=len(passed_inputs)
    num_failed=len(failed_inputs)
    is_too_hard=num_passed>num_failed*factor
    if is_too_hard:
        # make the current epsilon less 

        return [],[],epsilon


def select_from_images(num_iterations, images_tensor_array,epsilon_range):
    # do num_iterations where we adjust the epsilon
    current_epsilon=epsilon_range[0]
    passed_inputs=images_tensor_array
    failed_inputs=[]
    for i in range(num_iterations):
        passed_epsilons,failed_epsilons,new_eps=adjust_epsilon_according_to_passing_rates(passed_inputs,failed_inputs,epsilon_range,current_epsilon)

    


    return promising_images

    
def get_top_k_images(promising_images,images_naps,global_verified_epsilon):

    # these promising images are verified to be at least global_epsilon_verified
    epsilons=[0 for i in range(len(promising_images))]
    for i in range(len(promising_images)):
        
        input,nap=promising_images[i],images_naps[i]
        # I got the promising images
        # I should refine them 
        max_epsilon=max_epsilon_robustness_binary_search(input,nap,low=global_verified_epsilon)
        epsilons[i]=max_epsilon
    # then return the inputs sorted by their epsilons 
    





def export_robust_inputs():# as a json file that contains the input as tensor ,the robust epsilon without nap , the robust epsilon with nap




# make an explanation input out of it 
# that can be passed to the pipeline

def load_robust_input(json_file_path):
    # takes one of the json files and explains it
    


    return input_tensor,epsilon_with_nap







 # that is a very simple architechture 