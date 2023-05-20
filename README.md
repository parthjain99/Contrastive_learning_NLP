# Contrastive-Text-Generation  

This is our implementation of the paper [CoNT: Contrastive Neural Text Generation](https://arxiv.org/pdf/2205.14690.pdf) for the CS533 Class Project.

Contrastive learning has gained significant at- tention in the field of text generation due to its ability to alleviate exposure bias. However, pre- vious approaches applying contrastive learning to text generation have not provided significant improvements in performance. Authors of the paper CoNT:Contrastive Neural Text Genera- tion framework addresses this issue by intro- ducing strategies in three key aspects of Con- trastive Learning: selecting in-batch contrastive examples, using a contrastive loss, and infer- ence with a learned similarity function. We evaluate CoNT on two tasks - common sense generation on the Common-Gen dataset and text summarization on the X-Sum dataset - and try to replicate some of the results achieved by the authors of the original paper. Our experi- ments demonstrate that CoNT is a promising framework for improving text generation per- formance.


## Results

### Results on Summarization task
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"><span style="text-decoration:none;color:#000;background-color:transparent">Model</span></th>
    <th class="tg-c3ow"><span style="text-decoration:none">ROUGE</span><br>-1</th>
    <th class="tg-c3ow"><span style="text-decoration:none;color:#000;background-color:transparent">ROUGE</span><br><span style="color:#000">-2</span></th>
    <th class="tg-c3ow"><span style="text-decoration:none">ROUGE</span><br>-L</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">T5-small</span><br></td>
    <td class="tg-c3ow">36.10</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">14.72</span></td>
    <td class="tg-c3ow">29.16</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><span style="text-decoration:none;color:#000;background-color:transparent">T5-Naive CL</span></td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">36.34</span></td>
    <td class="tg-c3ow"><span style="text-decoration:none;color:#000;background-color:transparent">14.81</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">29.41</span></td>
  </tr>
  <tr>
    <td class="tg-qaub">T5-CONT</td>
    <td class="tg-qaub">39.66 <br></td>
    <td class="tg-qaub"><span style="font-weight:400;font-style:normal;text-decoration:none">16.96</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">31.86</span></td>
  </tr>
  <tr>
    <td class="tg-ncgp">Reproduced <br>Results</td>
    <td class="tg-ncgp"><span style="font-weight:400;font-style:normal;text-decoration:none">30.48</span></td>
    <td class="tg-ncgp">8.36</td>
    <td class="tg-ncgp"><span style="font-weight:400;font-style:normal;text-decoration:none">21.11</span><br></td>
  </tr>
</tbody>
</table>

###Results on Common Sense Generation task
<table class="tg">
<thead>
  <tr>
    <th class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">Model</span></th>
    <th class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">BLEU-3</span></th>
    <th class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">BLEU-4</span></th>
    <th class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">ROUGE -L</span></th>
    <th class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">METEOR</span></th>
    <th class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">CIDEr</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">T5-base</span></td>
    <td class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">28.76</span></td>
    <td class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">18.54</span></td>
    <td class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">34.56</span></td>
    <td class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">23.94</span></td>
    <td class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">9.4</span></td>
  </tr>
  <tr>
    <td class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">T5-1arge</span></td>
    <td class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">43.01</span></td>
    <td class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">31.96</span></td>
    <td class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">42.75</span></td>
    <td class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">31.12</span></td>
    <td class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">15.13</span></td>
  </tr>
  <tr>
    <td class="tg-wr11"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">T5-base-CONT</span></td>
    <td class="tg-wr11"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">42.6</span></td>
    <td class="tg-wr11"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">31.42</span></td>
    <td class="tg-za14"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">43.15</span></td>
    <td class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">32.05</span></td>
    <td class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">15.96</span></td>
  </tr>
  <tr>
    <td class="tg-wxen"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">T5-base-CONT (Reproduced)</span></td>
    <td class="tg-wxen"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">29.3</span></td>
    <td class="tg-wxen"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">20.6</span></td>
    <td class="tg-wxen"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">49.8</span></td>
    <td class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">28.9</span></td>
    <td class="tg-7zrl"><span style="font-weight:400;font-style:normal;text-decoration:none;color:black">12.67</span></td>
  </tr>
</tbody>
</table>

## Example Output for Commen Gen Task

<img src="https://github.com/parthjain99/Contrastive_learning_NLP/blob/main/Example.png">


