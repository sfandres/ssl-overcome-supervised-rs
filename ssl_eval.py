import torch
from torchinfo import summary
from finetuning_trainer import Trainer
import os


def linear_eval_backbone(
    curr_epoch,
    epochs,
    backbone,
    in_features,
    out_features,
    dataloader,
    batch_size,
    local_rank,
    paths,
    args,
    general_name,
    input_size: int = 224,
    verbose: bool = False,
    dropout: bool = False
):

    # Define your model.
    model = torch.nn.Sequential(
        backbone,
        torch.nn.Flatten(),
    )

    # Add dropout layer if the dropout argument is passed.
    if dropout:
        model.add_module('dropout', torch.nn.Dropout(p=dropout, inplace=True))
        print(f'Dropout layer added: {model.dropout}')
    else:
        print('No dropout layer')

    # Add the final linear layer
    model.add_module(
        'linear',
        torch.nn.Linear(in_features=in_features,
                        out_features=out_features,
                        bias=True)
    )

    # Get the number of input features to the layer.
    print(f'New final fully-connected layer: {model[-1]}')

    # Freezing all the network if chosen.
    for param in model.parameters():
        param.requires_grad = False
    for param in model[-1].parameters():
        param.requires_grad = True
    print('Linear probing adjusted')

    # Show model structure.
    if verbose:
        print(summary(
            model,
            input_size=(batch_size, 3, input_size, input_size),
            device=local_rank)
        )

    # Configure the loss.
    loss_fn = torch.nn.CrossEntropyLoss()
    print(f'Loss: {loss_fn}')

    # Configure the optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  #, momentum=0.9)

    # Training.
    args.task_name = 'multiclass'
    args.transfer_learning = 'LP'
    trainer = Trainer(
        model,
        dataloader,
        loss_fn,
        optimizer,
        save_every=10,
        snapshot_path=os.path.join(paths['snapshots'], f'head_{general_name}.pt'),
        csv_path=os.path.join(paths['csv_results'], f'head_{general_name}_e={curr_epoch}.csv'),
        distributed=False,
        lightly_train=True,
        ignore_ckpts=True
    )
    trainer.train(epochs, args, test=True, save_csv=True)
