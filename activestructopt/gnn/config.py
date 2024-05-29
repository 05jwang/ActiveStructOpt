torchmd_config = {
 'trainer': 'property',
 'task': {'identifier': 'my_train_job',
  'parallel': False,
  'save_dir': None,
  'continue_job': False,
  'load_training_state': False,
  'checkpoint_path': None,
  'write_output': [],
  'use_amp': True,
  'run_mode': 'train',
  'model_save_frequency': -1},
 'model': {
   'name': 'torchmd_etEarly',
   'hidden_channels': 64,
   'num_filters': 128,
   'num_layers': 6,
   'num_rbf': 50,
   'rbf_type': 'expnorm',
   'trainable_rbf': True,
   'activation': 'silu',
   'attn_activation': 'silu',
   'num_heads': 8,
   'distance_influence': 'both',
   'neighbor_embedding': True,
   'cutoff_lower': 0.0,
   'cutoff_upper': 8.0,
   'max_z': 100,
   'max_num_neighbors': 32,
   'aggr': 'add',
   'num_post_layers': 2,
   'post_hidden_channels': 128,
   'pool': 'global_mean_pool',
   'otf_edge_index': False,
   'otf_edge_attr': False,
   'otf_node_attr': False,
   'model_ensemble': 1,
   'gradient': False,
 },
 'optim': {'max_epochs': 500,
  'max_checkpoint_epochs': 0,
  'lr': 0.001,
  'loss': {'loss_type': 'TorchLossWrapper',
   'loss_args': {'loss_fn': 'l1_loss'}},
  'clip_grad_norm': 10,
  'batch_size': 1000,
  'optimizer': {'optimizer_type': 'AdamW', 'optimizer_args': {}},
  'scheduler': {'scheduler_type': 'ReduceLROnPlateau',
   'scheduler_args': {'mode': 'min',
    'factor': 0.8,
    'patience': 10,
    'min_lr': 1e-05,
    'threshold': 0.0002}},
  'verbosity': 5,
  'batch_tqdm': False},
 'dataset': {'name': 'test_data',
  'processed': False,
  'dataset_device': 'cpu',
  'target_path': None,
  'pt_path': 'data/',
  'prediction_level': 'graph',
  'transforms': [{'name': 'GetY', 'args': {'index': -1}, 'otf': True}],
  'data_format': 'json',
  'additional_attributes': None,
  'verbose': True,
  'preprocess_params': {'edge_calc_method': 'ocp',
   'preprocess_edges': True,
   'preprocess_edge_features': True,
   'preprocess_node_features': True,
   'cutoff_radius': 8.0,
   'n_neighbors': 250,
   'num_offsets': 2,
   'node_dim': 100,
   'edge_dim': 50,
   'prediction_level': 'graph',
   'self_loop': True,
   'node_representation': 'onehot',
   'all_neighbors': True}},
 'submit': None,
 'aso_params': {'max_forward_calls': 100,
  'dataset': {'N': 30, 
   'k': 5, 
   'perturbrmin': 0.0, 
   'perturbrmax': 1.0, 
   'split': 1/3, 
   'device': 'cuda', 
   'seed': 0},
  'train': {'finetune_epochs': 500, 
   'lr_reduction': 1.0},
  'opt': {'args': {'starts': 128, 
   'iters_per_start': 100, 
   'lr': 0.01}, 
   'switch_profiles': [20],
   'profiles': [{'obj_func': 'ucb_obj', 'obj_args': {'λ': 1.0},}, 
   {'obj_func': 'mse_obj', 'obj_args': {}}],
   }}}
