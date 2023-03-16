cd ..

echo "Creating a folder to store the datasets..."
mkdir datasets
echo "Completed!"

echo "Creating a folder to store the resulting figures..."
mkdir output
cd output/
mkdir figures
cd figures/
mkdir 0_history_log
mkdir simsiam
mkdir simclr
mkdir barlowtwins
mkdir finetuning
echo "Completed!"

cd ..

echo "Creating a folder to store the model checkpoints..."
mkdir checkpoints
cd checkpoints/
mkdir 0_history_log
mkdir simsiam
mkdir simclr
mkdir barlowtwins
mkdir finetuning
echo "Completed!"
