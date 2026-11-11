#!/usr/bin/env python3
"""Build Waymark AWS Reference Architecture in the official AWS Whitepaper style.

Adheres strictly to the user reference image (Amazon MWAA Architecture style):
- Clean navy title header
- Outer pink/magenta boundary container (AWS Region ap-southeast-2) with Region badge
- Inner green containers (Customer / Service VPCs) with VPC badges
- Inner solid orange service containers (Storage, Query, ML Training, Batch Scoring)
- White-on-orange compute icons and official AWS icons
- Purple VPCE and red Security Group integration interfaces
- Bottom horizontal backbone line with vertical drop lines to foundational AWS service icons
- Strictly orthogonal right-angled lines (rounded=0, curved=0)
- ZERO plan, phase, milestone, or stage terminology
"""
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def build_aws_reference_drawio():
    root = ET.Element("mxfile", host="Electron", agent="drawio-skill", version="24.0.0")
    diagram = ET.SubElement(
        root,
        "diagram",
        id="waymark-aws-mwaa-style",
        name="Waymark AWS Architecture",
    )
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1600",
        dy="1000",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth="1600",
        pageHeight="1020",
        math="0",
        shadow="0",
    )
    root_cell = ET.SubElement(model, "root")

    # Base cells
    ET.SubElement(root_cell, "mxCell", id="0")
    ET.SubElement(root_cell, "mxCell", id="1", parent="0")

    # 1. Main Title Header (Official AWS navy bold)
    title_cell = ET.SubElement(
        root_cell,
        "mxCell",
        id="arch_title",
        value="<font style='font-size: 30px; font-weight: 800; color: #0F2042; font-family: Segoe UI, Helvetica, Arial, sans-serif;'>Waymark Fulfillment Batch Architecture</font>",
        style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(title_cell, "mxGeometry", x="60", y="30", width="800", height="50", as_="geometry")
    geo.set("as", "geometry")

    # 2. Outer AWS Region Container (Official Magenta/Pink Border)
    region_box = ET.SubElement(
        root_cell,
        "mxCell",
        id="box_region",
        value="",
        style="rounded=1;arcSize=3;dashed=0;strokeColor=#D81B60;fillColor=none;strokeWidth=1.5;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(region_box, "mxGeometry", x="60", y="90", width="1480", height="610", as_="geometry")
    geo.set("as", "geometry")

    # Region Badge (Top Right of Outer Box)
    region_badge = ET.SubElement(
        root_cell,
        "mxCell",
        id="badge_region",
        value="",
        style="sketch=0;outlineConnect=0;fontColor=#232F3E;fillColor=#D81B60;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=10;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.region;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(region_badge, "mxGeometry", x="1480", y="95", width="44", height="44", as_="geometry")
    geo.set("as", "geometry")

    # Region Label (Top Left of Outer Box)
    region_lbl = ET.SubElement(
        root_cell,
        "mxCell",
        id="lbl_region",
        value="<font style='font-size: 13px; font-weight: bold; color: #D81B60;'>AWS Cloud: Region ap-southeast-2 (Sydney) | Account: 149270211548</font>",
        style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(region_lbl, "mxGeometry", x="75", y="95", width="500", height="25", as_="geometry")
    geo.set("as", "geometry")

    # 3. Left Green Sub-Container: Data Lake & Analytics VPC
    vpc_lake = ET.SubElement(
        root_cell,
        "mxCell",
        id="vpc_lake",
        value="",
        style="rounded=1;arcSize=3;dashed=0;strokeColor=#2E7D32;fillColor=#FFFFFF;strokeWidth=1.5;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(vpc_lake, "mxGeometry", x="80", y="125", width="680", height="555", as_="geometry")
    geo.set("as", "geometry")

    # VPC Badge + Label (Top Left of Lake VPC)
    vpc_lake_badge = ET.SubElement(
        root_cell,
        "mxCell",
        id="badge_vpc_lake",
        value="",
        style="sketch=0;outlineConnect=0;fontColor=#232F3E;fillColor=#2E7D32;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=10;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.vpc;",
        vertex="1",
        parent="vpc_lake",
    )
    geo = ET.SubElement(vpc_lake_badge, "mxGeometry", x="15", y="12", width="28", height="28", as_="geometry")
    geo.set("as", "geometry")

    vpc_lake_lbl = ET.SubElement(
        root_cell,
        "mxCell",
        id="lbl_vpc_lake",
        value="<b>Analytics & Storage Subnet</b>",
        style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=13;fontColor=#2E7D32;",
        vertex="1",
        parent="vpc_lake",
    )
    geo = ET.SubElement(vpc_lake_lbl, "mxGeometry", x="50", y="12", width="250", height="28", as_="geometry")
    geo.set("as", "geometry")

    # Orange Box 1 (Inside Lake VPC): Medallion Data Lake
    box_lake = ET.SubElement(
        root_cell,
        "mxCell",
        id="box_lake",
        value="",
        style="rounded=1;arcSize=3;dashed=0;strokeColor=#E65100;fillColor=#FFFFFF;strokeWidth=2;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;",
        vertex="1",
        parent="vpc_lake",
    )
    geo = ET.SubElement(box_lake, "mxGeometry", x="25", y="50", width="630", height="215", as_="geometry")
    geo.set("as", "geometry")

    box_lake_title = ET.SubElement(
        root_cell,
        "mxCell",
        id="title_lake",
        value="<b>Amazon S3 Medallion Lake</b>",
        style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=13;fontColor=#1E293B;",
        vertex="1",
        parent="box_lake",
    )
    geo = ET.SubElement(box_lake_title, "mxGeometry", x="10", y="180", width="610", height="25", as_="geometry")
    geo.set("as", "geometry")

    # Three S3 Tier Blocks
    tier_bronze = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_bronze",
        value="<b>Bronze Lake</b><br/><font color='#6B7280' style='font-size:10px;'>Raw Olist Data<br/>Source Manifests</font>",
        style="sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor=#E65100;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.s3;",
        vertex="1",
        parent="box_lake",
    )
    geo = ET.SubElement(tier_bronze, "mxGeometry", x="55", y="40", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    tier_silver = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_silver",
        value="<b>Silver Lake</b><br/><font color='#6B7280' style='font-size:10px;'>Cleaned Tables<br/>Key Reconciled</font>",
        style="sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor=#E65100;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.s3;",
        vertex="1",
        parent="box_lake",
    )
    geo = ET.SubElement(tier_silver, "mxGeometry", x="283", y="40", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    tier_gold = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_gold",
        value="<b>Gold Lake</b><br/><font color='#6B7280' style='font-size:10px;'>Fact & Dim Models<br/>Point-in-Time Features</font>",
        style="sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor=#E65100;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.s3;",
        vertex="1",
        parent="box_lake",
    )
    geo = ET.SubElement(tier_gold, "mxGeometry", x="510", y="40", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    # Orange Box 2 (Inside Lake VPC): Query & Transformation Engine
    box_query = ET.SubElement(
        root_cell,
        "mxCell",
        id="box_query",
        value="",
        style="rounded=1;arcSize=3;dashed=0;strokeColor=#E65100;fillColor=#FFFFFF;strokeWidth=2;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;",
        vertex="1",
        parent="vpc_lake",
    )
    geo = ET.SubElement(box_query, "mxGeometry", x="25", y="295", width="630", height="235", as_="geometry")
    geo.set("as", "geometry")

    box_query_title = ET.SubElement(
        root_cell,
        "mxCell",
        id="title_query",
        value="<b>Serverless Metadata & SQL Query Engine</b>",
        style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=13;fontColor=#1E293B;",
        vertex="1",
        parent="box_query",
    )
    geo = ET.SubElement(box_query_title, "mxGeometry", x="10", y="195", width="610", height="25", as_="geometry")
    geo.set("as", "geometry")

    node_glue = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_glue",
        value="<b>AWS Glue Catalog</b><br/><font color='#6B7280' style='font-size:10px;'>Schema Registry<br/>Python Shell (0.0625 DPU)</font>",
        style="sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor=#8C4FFF;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.glue;",
        vertex="1",
        parent="box_query",
    )
    geo = ET.SubElement(node_glue, "mxGeometry", x="135", y="45", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    node_athena = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_athena",
        value="<b>Amazon Athena v3</b><br/><font color='#6B7280' style='font-size:10px;'>Interactive Analytics<br/>1 GB Query Cutoff Guard</font>",
        style="sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor=#8C4FFF;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.athena;",
        vertex="1",
        parent="box_query",
    )
    geo = ET.SubElement(node_athena, "mxGeometry", x="430", y="45", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    # 4. Right Green Sub-Container: Machine Learning & Batch VPC
    vpc_ml = ET.SubElement(
        root_cell,
        "mxCell",
        id="vpc_ml",
        value="",
        style="rounded=1;arcSize=3;dashed=0;strokeColor=#2E7D32;fillColor=#FFFFFF;strokeWidth=1.5;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(vpc_ml, "mxGeometry", x="840", y="125", width="680", height="555", as_="geometry")
    geo.set("as", "geometry")

    vpc_ml_badge = ET.SubElement(
        root_cell,
        "mxCell",
        id="badge_vpc_ml",
        value="",
        style="sketch=0;outlineConnect=0;fontColor=#232F3E;fillColor=#2E7D32;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=10;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.vpc;",
        vertex="1",
        parent="vpc_ml",
    )
    geo = ET.SubElement(vpc_ml_badge, "mxGeometry", x="15", y="12", width="28", height="28", as_="geometry")
    geo.set("as", "geometry")

    vpc_ml_lbl = ET.SubElement(
        root_cell,
        "mxCell",
        id="lbl_vpc_ml",
        value="<b>Machine Learning & Inference Subnet</b>",
        style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=13;fontColor=#2E7D32;",
        vertex="1",
        parent="vpc_ml",
    )
    geo = ET.SubElement(vpc_ml_lbl, "mxGeometry", x="50", y="12", width="280", height="28", as_="geometry")
    geo.set("as", "geometry")

    # Orange Box 3 (Inside ML VPC): Distributed Model Training Cluster
    box_train = ET.SubElement(
        root_cell,
        "mxCell",
        id="box_train",
        value="",
        style="rounded=1;arcSize=3;dashed=0;strokeColor=#E65100;fillColor=#FFFFFF;strokeWidth=2;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;",
        vertex="1",
        parent="vpc_ml",
    )
    geo = ET.SubElement(box_train, "mxGeometry", x="25", y="50", width="630", height="215", as_="geometry")
    geo.set("as", "geometry")

    box_train_title = ET.SubElement(
        root_cell,
        "mxCell",
        id="title_train",
        value="<b>Amazon SageMaker Spot Training (ml.m5.large)</b>",
        style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=13;fontColor=#1E293B;",
        vertex="1",
        parent="box_train",
    )
    geo = ET.SubElement(box_train_title, "mxGeometry", x="10", y="180", width="610", height="25", as_="geometry")
    geo.set("as", "geometry")

    node_m1 = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_m1",
        value="<b>M1 Delivery ETA</b><br/><font color='#6B7280' style='font-size:10px;'>LightGBM Quantiles<br/>Split-Conformal Bounds</font>",
        style="sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor=#01A88D;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.sagemaker_2;",
        vertex="1",
        parent="box_train",
    )
    geo = ET.SubElement(node_m1, "mxGeometry", x="135", y="40", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    node_m2 = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_m2",
        value="<b>M2 Delay Risk</b><br/><font color='#6B7280' style='font-size:10px;'>CatBoost Classifier<br/>Platt & SHAP Attribution</font>",
        style="sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor=#01A88D;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.sagemaker_2;",
        vertex="1",
        parent="box_train",
    )
    geo = ET.SubElement(node_m2, "mxGeometry", x="430", y="40", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    # Orange Box 4 (Inside ML VPC): Automated Batch Transform Pipeline
    box_batch = ET.SubElement(
        root_cell,
        "mxCell",
        id="box_batch",
        value="",
        style="rounded=1;arcSize=3;dashed=0;strokeColor=#E65100;fillColor=#FFFFFF;strokeWidth=2;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;",
        vertex="1",
        parent="vpc_ml",
    )
    geo = ET.SubElement(box_batch, "mxGeometry", x="25", y="295", width="630", height="235", as_="geometry")
    geo.set("as", "geometry")

    box_batch_title = ET.SubElement(
        root_cell,
        "mxCell",
        id="title_batch",
        value="<b>Batch Inference & Idempotent Orchestration</b>",
        style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=13;fontColor=#1E293B;",
        vertex="1",
        parent="box_batch",
    )
    geo = ET.SubElement(box_batch_title, "mxGeometry", x="10", y="195", width="610", height="25", as_="geometry")
    geo.set("as", "geometry")

    node_eb = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_eb",
        value="<b>EventBridge</b><br/><font color='#6B7280' style='font-size:10px;'>Daily Schedule<br/>Automated Trigger</font>",
        style="sketch=0;outlineConnect=0;fontColor=#232F3E;fillColor=#E7157B;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;pointerEvents=1;shape=mxgraph.aws4.eventbridge_scheduler;",
        vertex="1",
        parent="box_batch",
    )
    geo = ET.SubElement(node_eb, "mxGeometry", x="55", y="45", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    node_lambda = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_lambda",
        value="<b>Lambda Adapter</b><br/><font color='#6B7280' style='font-size:10px;'>Python 3.12 arm64<br/>_SUCCESS Guard</font>",
        style="sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor=#ED7100;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.lambda;",
        vertex="1",
        parent="box_batch",
    )
    geo = ET.SubElement(node_lambda, "mxGeometry", x="283", y="45", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    node_bt = ET.SubElement(
        root_cell,
        "mxCell",
        id="node_bt",
        value="<b>Batch Transform</b><br/><font color='#6B7280' style='font-size:10px;'>SageMaker Offline<br/>ml.m5.large Predictor</font>",
        style="sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor=#01A88D;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.sagemaker_2;",
        vertex="1",
        parent="box_batch",
    )
    geo = ET.SubElement(node_bt, "mxGeometry", x="510", y="45", width="64", height="64", as_="geometry")
    geo.set("as", "geometry")

    # 5. Inter-VPC Interfaces (Middle Channel between VPCs: x=760 to 840)
    # VPCE Icon 1 (Data Interface)
    vpce_data = ET.SubElement(
        root_cell,
        "mxCell",
        id="vpce_data",
        value="",
        style="sketch=0;outlineConnect=0;fontColor=#232F3E;fillColor=#8C4FFF;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=10;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.endpoint;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(vpce_data, "mxGeometry", x="782", y="235", width="36", height="36", as_="geometry")
    geo.set("as", "geometry")

    vpce_data_lbl = ET.SubElement(
        root_cell,
        "mxCell",
        id="lbl_vpce_data",
        value="<i>Training Data<br/>VPCE</i>",
        style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;fontColor=#4B5563;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(vpce_data_lbl, "mxGeometry", x="760", y="200", width="80", height="30", as_="geometry")
    geo.set("as", "geometry")

    # VPCE Icon 2 (Inference Result Interface)
    vpce_res = ET.SubElement(
        root_cell,
        "mxCell",
        id="vpce_res",
        value="",
        style="sketch=0;outlineConnect=0;fontColor=#232F3E;fillColor=#8C4FFF;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=10;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.endpoint;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(vpce_res, "mxGeometry", x="782", y="500", width="36", height="36", as_="geometry")
    geo.set("as", "geometry")

    vpce_res_lbl = ET.SubElement(
        root_cell,
        "mxCell",
        id="lbl_vpce_res",
        value="<i>Inference<br/>VPCE</i>",
        style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;fontColor=#4B5563;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(vpce_res_lbl, "mxGeometry", x="760", y="465", width="80", height="30", as_="geometry")
    geo.set("as", "geometry")

    # Security Group Badges on Borders
    sg_lake = ET.SubElement(
        root_cell,
        "mxCell",
        id="sg_lake",
        value="",
        style="sketch=0;outlineConnect=0;fontColor=#232F3E;fillColor=#DD344C;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=10;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.identity_and_access_management;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(sg_lake, "mxGeometry", x="746", y="350", width="28", height="28", as_="geometry")
    geo.set("as", "geometry")

    sg_ml = ET.SubElement(
        root_cell,
        "mxCell",
        id="sg_ml",
        value="",
        style="sketch=0;outlineConnect=0;fontColor=#232F3E;fillColor=#DD344C;strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=10;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.identity_and_access_management;",
        vertex="1",
        parent="1",
    )
    geo = ET.SubElement(sg_ml, "mxGeometry", x="826", y="350", width="28", height="28", as_="geometry")
    geo.set("as", "geometry")

    # 6. Orthogonal Connectors (Strict Right Angles, Zero Curves)
    base_ortho = "edgeStyle=orthogonalEdgeStyle;rounded=0;curved=0;orthogonalLoop=1;jettySize=auto;html=1;"

    # Lake Internal Flow: Bronze -> Silver -> Gold
    e_bs = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_bronze_silver",
        value="",
        style=f"{base_ortho}strokeColor=#4B5563;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        edge="1",
        parent="box_lake",
        source="node_bronze",
        target="node_silver",
    )
    ET.SubElement(e_bs, "mxGeometry", relative="1", as_="geometry").set("as", "geometry")

    e_sg = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_silver_gold",
        value="",
        style=f"{base_ortho}strokeColor=#4B5563;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        edge="1",
        parent="box_lake",
        source="node_silver",
        target="node_gold",
    )
    ET.SubElement(e_sg, "mxGeometry", relative="1", as_="geometry").set("as", "geometry")

    # Lake to Query Flow: Gold -> Athena
    e_ga = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_gold_athena",
        value="",
        style=f"{base_ortho}strokeColor=#4B5563;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;",
        edge="1",
        parent="vpc_lake",
        source="box_lake",
        target="box_query",
    )
    ET.SubElement(e_ga, "mxGeometry", relative="1", as_="geometry").set("as", "geometry")

    # Query internal: Glue -> Athena
    e_ga2 = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_glue_athena",
        value="Catalog & Views",
        style=f"{base_ortho}strokeColor=#4B5563;strokeWidth=1.5;fontSize=10;fontColor=#4B5563;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        edge="1",
        parent="box_query",
        source="node_glue",
        target="node_athena",
    )
    ET.SubElement(e_ga2, "mxGeometry", relative="1", as_="geometry").set("as", "geometry")

    # Inter-VPC Training Flow: Athena -> SageMaker (Through VPCE Data)
    e_train = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_train_feed",
        value="",
        style=f"{base_ortho}strokeColor=#059669;strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        edge="1",
        parent="1",
        source="vpc_lake",
        target="vpce_data",
    )
    g_tr = ET.SubElement(e_train, "mxGeometry", relative="1", as_="geometry")
    g_tr.set("as", "geometry")
    pts_tr = ET.SubElement(g_tr, "Array", as_="points")
    pts_tr.set("as", "points")
    ET.SubElement(pts_tr, "mxPoint", x="760", y="253")

    e_train2 = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_train_feed2",
        value="",
        style=f"{base_ortho}strokeColor=#059669;strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        edge="1",
        parent="1",
        source="vpce_data",
        target="vpc_ml",
    )
    g_tr2 = ET.SubElement(e_train2, "mxGeometry", relative="1", as_="geometry")
    g_tr2.set("as", "geometry")
    pts_tr2 = ET.SubElement(g_tr2, "Array", as_="points")
    pts_tr2.set("as", "points")
    ET.SubElement(pts_tr2, "mxPoint", x="840", y="253")

    # Training to Batch: M1/M2 Models Deploy
    e_mb = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_model_deploy",
        value="Deploy Model Artifacts",
        style=f"{base_ortho}strokeColor=#059669;strokeWidth=1.5;fontSize=10;fontColor=#065F46;exitX=0.5;exitY=1;entryX=0.5;entryY=0;",
        edge="1",
        parent="vpc_ml",
        source="box_train",
        target="box_batch",
    )
    ET.SubElement(e_mb, "mxGeometry", relative="1", as_="geometry").set("as", "geometry")

    # Batch internal: EventBridge -> Lambda -> Batch Transform
    e_ebl = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_eb_lambda",
        value="Trigger",
        style=f"{base_ortho}strokeColor=#E65100;strokeWidth=1.5;fontSize=10;fontColor=#92400E;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        edge="1",
        parent="box_batch",
        source="node_eb",
        target="node_lambda",
    )
    ET.SubElement(e_ebl, "mxGeometry", relative="1", as_="geometry").set("as", "geometry")

    e_lbt = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_lambda_bt",
        value="CreateTransformJob",
        style=f"{base_ortho}strokeColor=#E65100;strokeWidth=1.5;fontSize=10;fontColor=#92400E;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        edge="1",
        parent="box_batch",
        source="node_lambda",
        target="node_bt",
    )
    ET.SubElement(e_lbt, "mxGeometry", relative="1", as_="geometry").set("as", "geometry")

    # Inter-VPC Inference Return Flow: Batch Transform -> VPCE Res -> S3 Gold Lake
    e_ret = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_pred_return",
        value="",
        style=f"{base_ortho}dashed=1;strokeColor=#2563EB;strokeWidth=2;exitX=0;exitY=0.5;entryX=1;entryY=0.5;",
        edge="1",
        parent="1",
        source="vpc_ml",
        target="vpce_res",
    )
    g_ret = ET.SubElement(e_ret, "mxGeometry", relative="1", as_="geometry")
    g_ret.set("as", "geometry")
    pts_ret = ET.SubElement(g_ret, "Array", as_="points")
    pts_ret.set("as", "points")
    ET.SubElement(pts_ret, "mxPoint", x="840", y="518")

    e_ret2 = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_pred_return2",
        value="Publish Predictions & _SUCCESS",
        style=f"{base_ortho}dashed=1;strokeColor=#2563EB;strokeWidth=2;fontSize=10;fontColor=#1E40AF;exitX=0;exitY=0.5;entryX=1;entryY=0.5;",
        edge="1",
        parent="1",
        source="vpce_res",
        target="vpc_lake",
    )
    g_ret2 = ET.SubElement(e_ret2, "mxGeometry", relative="1", as_="geometry")
    g_ret2.set("as", "geometry")
    pts_ret2 = ET.SubElement(g_ret2, "Array", as_="points")
    pts_ret2.set("as", "points")
    ET.SubElement(pts_ret2, "mxPoint", x="760", y="518")

    # 7. Bottom Foundation Services Row (The official AWS Whitepaper bar)
    # Vertical connector dropping down from main region box
    e_down = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_backbone_down",
        value="",
        style="edgeStyle=orthogonalEdgeStyle;rounded=0;curved=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#64748B;strokeWidth=1.5;endArrow=none;exitX=0.5;exitY=1;",
        edge="1",
        parent="1",
        source="box_region",
    )
    g_down = ET.SubElement(e_down, "mxGeometry", relative="1", as_="geometry")
    g_down.set("as", "geometry")
    pts_down = ET.SubElement(g_down, "Array", as_="points")
    pts_down.set("as", "points")
    ET.SubElement(pts_down, "mxPoint", x="800", y="700")
    ET.SubElement(pts_down, "mxPoint", x="800", y="735")

    # Horizontal backbone line
    e_spine = ET.SubElement(
        root_cell,
        "mxCell",
        id="e_backbone_spine",
        value="",
        style="edgeStyle=orthogonalEdgeStyle;rounded=0;curved=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#64748B;strokeWidth=1.5;endArrow=none;",
        edge="1",
        parent="1",
    )
    g_spine = ET.SubElement(e_spine, "mxGeometry", relative="1", as_="geometry")
    g_spine.set("as", "geometry")
    pts_spine = ET.SubElement(g_spine, "Array", as_="points")
    pts_spine.set("as", "points")
    ET.SubElement(pts_spine, "mxPoint", x="135", y="735")
    ET.SubElement(pts_spine, "mxPoint", x="1465", y="735")

    # 9 Foundational Service Icons along the bottom
    services = [
        ("svc_cw", "Amazon CloudWatch", "Log Management & Metrics", "mxgraph.aws4.cloudwatch_2", "#E7157B", 135),
        ("svc_s3", "Amazon S3", "Object Lake Storage", "mxgraph.aws4.s3", "#7AA116", 301),
        ("svc_glue", "AWS Glue", "Unified Data Catalog", "mxgraph.aws4.glue", "#8C4FFF", 467),
        ("svc_athena", "Amazon Athena", "Serverless SQL Engine", "mxgraph.aws4.athena", "#8C4FFF", 633),
        ("svc_sm", "Amazon SageMaker", "ML Training & Inference", "mxgraph.aws4.sagemaker_2", "#01A88D", 800),
        ("svc_lambda", "AWS Lambda", "Event-Driven Compute", "mxgraph.aws4.lambda", "#ED7100", 967),
        ("svc_eb", "Amazon EventBridge", "Scheduled Event Router", "mxgraph.aws4.eventbridge_scheduler", "#E7157B", 1133),
        ("svc_iam", "AWS IAM", "Least-Privilege Roles", "mxgraph.aws4.identity_and_access_management", "#DD344C", 1299),
        ("svc_budgets", "AWS Budgets", "Cost Anomaly & Alerts", "mxgraph.aws4.budgets_2", "#7AA116", 1465),
    ]

    for sid, sname, sdesc, sicon, sfill, sx in services:
        # Drop line from backbone
        drop_line = ET.SubElement(
            root_cell,
            "mxCell",
            id=f"drop_{sid}",
            value="",
            style="edgeStyle=orthogonalEdgeStyle;rounded=0;curved=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#64748B;strokeWidth=1.5;endArrow=block;endFill=1;",
            edge="1",
            parent="1",
        )
        g_dl = ET.SubElement(drop_line, "mxGeometry", relative="1", as_="geometry")
        g_dl.set("as", "geometry")
        pts_dl = ET.SubElement(g_dl, "Array", as_="points")
        pts_dl.set("as", "points")
        ET.SubElement(pts_dl, "mxPoint", x=str(sx), y="735")
        ET.SubElement(pts_dl, "mxPoint", x=str(sx), y="760")

        # Service Icon
        s_cell = ET.SubElement(
            root_cell,
            "mxCell",
            id=sid,
            value=f"<b>{sname}</b><br/><font color='#6B7280' style='font-size:10px;'>{sdesc}</font>",
            style=f"sketch=0;points=[[0,0,0],[0.25,0,0],[0.5,0,0],[0.75,0,0],[1,0,0],[0,1,0],[0.25,1,0],[0.5,1,0],[0.75,1,0],[1,1,0],[0,0.25,0],[0,0.5,0],[0,0.75,0],[1,0.25,0],[1,0.5,0],[1,0.75,0]];outlineConnect=0;fontColor=#232F3E;fillColor={sfill};strokeColor=#ffffff;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon={sicon};",
            vertex="1",
            parent="1",
        )
        geo_s = ET.SubElement(s_cell, "mxGeometry", x=str(sx - 28), y="765", width="56", height="56", as_="geometry")
        geo_s.set("as", "geometry")

    # Format XML string
    xml_str = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_str)
    pretty_xml = parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    out_path = os.path.abspath("data/aws_setup/waymark_aws_architecture.drawio")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    print(f"Created: {out_path}")
    return out_path


if __name__ == "__main__":
    build_aws_reference_drawio()
