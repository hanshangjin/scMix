## Datasets and Preprocessing

The datasets used in this project are summarized below.

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Species</th>
      <th># cells</th>
      <th># TPs</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Zebrafish Embryo</td>
      <td><i>Danio rerio</i></td>
      <td>38 731</td>
      <td>12</td>
      <td>
        Single Cell Portal 
        (<a href="https://singlecell.broadinstitute.org/single_cell/study/SCP162/single-cell-reconstruction-of-developmental-trajectories-during-zebrafish-embryogenesis">SCP162</a>)
      </td>
    </tr>
    <tr>
      <td>Drosophila</td>
      <td><i>Drosophila melanogaster</i></td>
      <td>27 386</td>
      <td>11</td>
      <td>
        DEAP website 
        (<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190147">GSE190147</a>)
      </td>
    </tr>
    <tr>
      <td>Schiebinger2019</td>
      <td><i>Mus musculus</i></td>
      <td>236 285</td>
      <td>19</td>
      <td>
        Broad Institute 
        (<a href="https://broadinstitute.github.io/wot/tutorial/">Wot</a>)
      </td>
    </tr>
    <tr>
      <td>Veres</td>
      <td><i>Homo sapiens</i></td>
      <td>51 274</td>
      <td>8</td>
      <td>
        GEO 
        (<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114412">GSE114412</a>)
      </td>
    </tr>
  </tbody>
</table>

For **Zebrafish Embryo**, **Drosophila**, and **Schiebinger2019**, we followed the preprocessing pipeline provided in [scNODE](https://github.com/rsinghlab/scNODE).

For **Veres**, the preprocessing procedure was based on [PI-SDE](https://github.com/QiJiang-QJ/PI-SDE), with several adjustments to the original procedure.
