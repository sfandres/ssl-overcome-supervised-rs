echo "Creating a folder to store the resulting images..."
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
mkdir pytorch_models
cd pytorch_models/
mkdir 0_history_log
mkdir simsiam
mkdir simclr
mkdir barlowtwins
mkdir finetuning
echo "Completed!"
