import os
from collections import Counter

import pandas as pd
import numpy as np

submissions_dir = 'e_selection'
submissions = os.listdir(submissions_dir)

print(submissions)

sbm_weigths = {
    '2005_submission_ens_3.csv': 2005,
    '2022_submission_NAZARII_FEATURES.csv': 2022,
    '2065_submission_finals.csv': 2065,
    '2126_submission_finals.csv': 2126,
    '2134_submission_finals.csv': 2134,
    '2148_submission_ens_3_finals.csv': 2148,
    '2210_submission_finals.csv': 2210,
    '2218_submission_finals.csv': 2218
}

targets = []
files = []
for submission in submissions:
    files.append(submission)
    sbmsn = pd.read_csv(os.path.join(submissions_dir, submission))
    sbmsn = sbmsn.sort_values('EmployeeID')
    targets.append(np.array(sbmsn['target']))
    ids = sbmsn['EmployeeID']

final_targets = []
for j in range(len(targets[0])):
    trg = [0, 0]
    for t in range(len(targets)):
        trg[targets[t][j]] += sbm_weigths[files[t]]
    final_targets.append(np.argmax(trg))

submission_df = pd.DataFrame(
            {'EmployeeID': ids,
             'target': list(final_targets)}
        )
submission_df.to_csv('FINAL_SUBMISSION.csv', index=False)
