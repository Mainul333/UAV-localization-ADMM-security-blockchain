UAV revision - blockchain experiments (one command)
====================================================

1. Unzip this folder INSIDE your "6 Nodes" folder, next to geth.exe:

   Geth Using Installed Software\
     6 Nodes\
       geth.exe, bootnode.exe, boot.key, genesis.json, node1 ... node6, command-*.bat
       uav_blockchain_run\          <- this folder
         RUN_ME.sh                  <- the only file you run

2. Close every other Geth window (for example the mainnet one) and do NOT
   double-click any command-*.bat file. The script starts the network itself.

3. Open Git Bash inside uav_blockchain_run (right-click the folder ->
   "Open Git Bash here") and run:

       bash RUN_ME.sh

   It backs up node1..node6 once, starts the boot node and the six nodes, waits
   for blocks, prepares the keys and the contract, runs all experiments
   (about 30-60 minutes), stops the nodes and creates
   blockchain_results_<date>.zip in this folder.

4. Upload blockchain_results_<date>.zip.

If it stops half way:  RESUME=1 bash RUN_ME.sh   (finished steps are skipped)
If Windows Firewall asks about geth.exe or bootnode.exe, click "Allow".
The private/ folder holds keys and test accounts; it is never put in the zip.

Load test up to 1000 UAVs (optional, adds measurements for the revision)
------------------------------------------------------------------------
    bash RUN_LOADTEST.sh
Starts the network, lets 6, 25, 50, 100, 200, 500, and 1000 UAV accounts submit one
state record each (5 rounds per size), and measures per round:
  - latency of one iteration and client-observed TPS (records / time until the last receipt)
  - on-chain TPS (records / block-timestamp interval needed to include them)
  - blocks used, block size (bytes), and the exact size of each signed transaction (bytes)
Upload loadtest_results_<date>.zip. The 200 load-test accounts from the first run are
reused; 800 new ones are created, funded (0.2 ETH each), and registered.

Optimized load test (latency reduction)
---------------------------------------
    bash RESET_CHAIN.sh      (once; all Geth windows closed)
    bash RUN_FASTTEST.sh
RESET_CHAIN.sh starts a fresh chain on the same network: it moves each node's chain data
to nodeX/geth_backup_<date> (keys, passwords and node identities stay), writes
genesis_fast.json (= genesis.json with a 350 M block gas limit), and initializes the six
nodes with the hash state scheme. RUN_FASTTEST.sh then runs the nodes in archive mode,
so that an abrupt stop no longer rewinds the chain, deploys the contract again, and funds
and registers the test accounts automatically.
Same network, same smart contract, three changes that remove the bottlenecks of the
first load test:
  1. All six nodes run with --miner.gaslimit 350000000, so that the 1000 records of one
     iteration fit into one block (the default 30 M gas holds only 107 records). Geth
     raises the limit by about 0.1% per block: the first run waits about 40 minutes for
     the limit to rise; afterwards it stays in the chain.
  2. Every UAV signs its own record (emulated by signing before the timed window; the
     signing time per record is reported separately).
  3. Every record is sent to all six validators in batched requests, so that the next
     sealer always holds it (before, half of the blocks were empty).
Upload fasttest_results_<date>.zip. Settings: GAS_TARGET, GAS_WAIT_MAX, LOAD_SIZES,
LOAD_ROUNDS (environment variables).
Note: RUN_ME.sh and RUN_LOADTEST.sh do not set the gas target; if they are run later,
the signers slowly lower the limit back towards 30 M.
