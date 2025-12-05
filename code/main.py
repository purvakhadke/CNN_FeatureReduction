# main file runs all the code in order, so you dont have to run 1 file at a time (can go AFK)

files = [
        '0_prepare_data.py', 
        '1_resnet_classifier.py', 
        '2_transformer_classifier.py',
        '3_autoencoder_classifier.py',
        '4_compare_results.py'
]

for file in files:
    print(f"Running {file}")
    with open(file) as f:
        exec(f.read())

print("ALL DONE")
print("ALL DONE")
print("ALL DONE")
print("ALL DONE")